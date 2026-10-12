#pragma once
#include <atomic>
#include <exception>
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

struct MediaDeviceLimit {
    int device = -1;
    Bytes context_unaccounted = 0, safety_margin = 0;
};

struct MediaAdmissionEvidence {
    std::vector<DeviceBudget> budgets;
    PeerMatrix peers;
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
// failure must leave nothing behind. No runtime route selects this prerequisite.
class MediaDeviceOperations {
public:
    virtual ~MediaDeviceOperations() = default;
    virtual std::size_t allocation_alignment() const = 0;
    virtual Bytes free_memory(int device) = 0;
    virtual bool direct_peer(int from, int to) = 0;
    virtual void* allocate(int device, Bytes bytes, std::size_t alignment) = 0;
    virtual void free(int device, void*) = 0;
    virtual void synchronize(int device) = 0;
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

private:
    friend class MediaSession;
    friend class MediaSessionState;
    std::shared_ptr<MediaSessionState> state_;
    std::size_t slot_ = 0;
    MediaDeviceSpan layout_;
    std::string name_;
    void release() noexcept;
};

// Session and its views are orchestrated by one request-owner thread.
class MediaSession {
public:
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
    void finish();
    void abort(std::exception_ptr first_error = {});

private:
    std::shared_ptr<MediaSessionState> state_;
};

}  // namespace kadan
