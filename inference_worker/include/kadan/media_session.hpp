#pragma once
#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <string_view>

#include "kadan/media_residency.hpp"

namespace kadan {
enum class DeviceDType {
    bf16,
    f16,
    f32,
    f64,
    i8,
    u8,
    u32
};

struct MediaEvent {
    int device = -1;
    std::uint64_t id = 0;
    auto operator<=>(const MediaEvent&) const = default;
};

struct MediaDeviceLimit {
    int device = -1;
    Bytes context_unaccounted = 0, safety_margin = 0;
};

struct MediaAdmissionEvidence {
    std::vector<DeviceBudget> budgets;
    PeerMatrix peers;
};

struct MediaTransferCounters {
    Bytes activation_upload = 0, activation_download = 0, local_activation = 0;
    Bytes input_upload = 0, weight_upload = 0, output_download = 0, peer_activation = 0,
          control_download = 0;
};

// Shared with WorkerCompute: even an unconfirmed cleanup blocks idle/park.
struct MediaSessionActivity {
    std::atomic_size_t owners = 0;
};

struct MediaDeviceSpan {
    void* data = nullptr;
    int device = -1;
    DeviceDType dtype = DeviceDType::f32;
    std::vector<std::size_t> dimensions, strides;
    Bytes bytes = 0;
    std::uint64_t generation = 0;
    Bytes offset = 0;
};

// CUDA and CPU lifecycle fakes implement the same device boundary. Allocation
// failure must leave nothing behind. Events remain valid until release_events.
// No cross-worker future waits: the request owner enqueues device event waits.
class MediaDeviceOperations {
public:
    virtual ~MediaDeviceOperations() = default;
    virtual std::size_t allocation_alignment() const = 0;
    virtual void* stream(int device) = 0;
    virtual Bytes free_memory(int device) = 0;
    virtual bool direct_peer(int from, int to) = 0;
    virtual void* allocate(int device, Bytes bytes, std::size_t alignment) = 0;
    virtual void free(int device, void*) = 0;
    virtual void wait(int device, MediaEvent) = 0;
    virtual MediaEvent record(int device) = 0;
    virtual void synchronize(int device) = 0;
    virtual void release_events(int device) = 0;
    virtual void upload(int device, void*, std::span<const std::byte>) = 0;
    virtual void download(int device, std::span<std::byte>, const void*) = 0;
    virtual void peer(int from, const void*, int to, void*, Bytes) = 0;
    virtual std::uint32_t read_status(int device, const void*) = 0;
};
class MediaSessionState;

class DeviceTensorView {
public:
    DeviceTensorView() = default;
    DeviceTensorView(DeviceTensorView&&) noexcept;
    DeviceTensorView& operator=(DeviceTensorView&&) noexcept;
    ~DeviceTensorView();
    DeviceTensorView(const DeviceTensorView&) = delete;
    DeviceTensorView& operator=(const DeviceTensorView&) = delete;
    const MediaDeviceSpan& layout() const;
    MediaEvent readiness() const;

private:
    friend class MediaSession;
    friend class MediaSessionState;
    std::shared_ptr<MediaSessionState> state_;
    std::size_t slot_ = 0;
    MediaDeviceSpan layout_;
    std::string name_;
    MediaEvent produced_;
    void release() noexcept;
};

// Session and its views are orchestrated by one request-owner thread.
class MediaSession {
public:
    using Launch = std::function<void(
        void* stream, std::span<const MediaDeviceSpan>, std::span<const MediaDeviceSpan>
    )>;
    MediaSession(
        std::shared_ptr<Resources>,
        std::shared_ptr<MediaSessionActivity>,
        std::unique_ptr<MediaDeviceOperations>,
        Workload,
        std::vector<MediaDeviceLimit>,
        const std::atomic_bool& cancel
    );
    MediaSession(MediaSession&&) noexcept = default;
    MediaSession& operator=(MediaSession&&) noexcept;
    ~MediaSession();
    MediaSession(const MediaSession&) = delete;
    MediaSession& operator=(const MediaSession&) = delete;

    // The actual consumed manifest and implemented workspace layout are required.
    // Planner failures propagate as typed values, before allocation/submission.
    std::optional<MediaPlanError> admit(const StageManifest&, const MediaShape&);
    const MediaResidencyPlan& plan() const;
    const MediaAdmissionEvidence& evidence() const;
    const MediaTransferCounters& transfers() const;
    void advance(MediaPhase);
    DeviceTensorView tensor(
        int device,
        std::string_view name,
        DeviceDType,
        std::vector<std::size_t> dimensions,
        std::vector<std::size_t> strides
    );
    DeviceTensorView weight(const WeightSlice&);
    DeviceTensorView rows(DeviceTensorView&, std::size_t first, std::size_t count);
    // Explicit host boundaries, synchronous for host-span lifetime. The ingress
    // caller supplies validated input; weight copies preserve original raw bytes.
    void upload_input(DeviceTensorView&, std::span<const std::byte>);
    void upload_weight(DeviceTensorView&, std::span<const std::byte>);
    void download_output(DeviceTensorView&, std::span<std::byte>);
    // Each gather cycle starts at column zero and fills contiguous column shards
    // in order. Restarting invalidates all prior coverage in the destination view.
    void gather(
        DeviceTensorView& source,
        DeviceTensorView& destination,
        std::size_t first_column
    );
    // Launchers must use this device's owned stream and write a zero/nonzero
    // uint32 status word. They preserve model-specific casts/reduction order.
    void launch(
        int device,
        std::span<DeviceTensorView* const> inputs,
        std::span<DeviceTensorView* const> outputs,
        const Launch&
    );
    // Every output and status must belong to the same completed launch event.
    // No produced tensor can become another operation's input before this check.
    void complete(std::span<DeviceTensorView* const> outputs, DeviceTensorView& status);
    void finish();
    void abort(std::exception_ptr first_error = {});

private:
    std::shared_ptr<MediaSessionState> state_;
    void upload(DeviceTensorView&, std::span<const std::byte>, bool weight);
};

}  // namespace kadan
