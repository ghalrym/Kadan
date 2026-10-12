#pragma once
#include <atomic>
#include <exception>
#include <memory>

#include "kadan/media_residency.hpp"

namespace kadan {


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
    void finish();
    void abort(std::exception_ptr first_error = {});

private:
    std::shared_ptr<MediaSessionState> state_;
};

}  // namespace kadan
