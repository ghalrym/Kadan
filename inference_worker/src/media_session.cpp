#include "kadan/media_session.hpp"

#include <algorithm>
#include <map>
#include <set>

#include "kadan/cleanup_error.hpp"

namespace kadan {
namespace {
Bytes add(Bytes a, Bytes b) {
    if (b > std::numeric_limits<Bytes>::max() - a)
        throw std::overflow_error("media_session_bytes");
    return a + b;
}

Bytes mul(Bytes a, Bytes b) {
    if (a && b > std::numeric_limits<Bytes>::max() / a)
        throw std::overflow_error("media_session_bytes");
    return a * b;
}

Bytes aligned(Bytes bytes, std::size_t alignment) {
    return mul(add(bytes, alignment - 1) / alignment, alignment);
}

std::atomic_uint64_t next_owner{1};

std::uint64_t owner_generation() {
    auto value = next_owner.load();
    do {
        if (value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("media_session_generation");
    } while (!next_owner.compare_exchange_weak(value, value + 1));
    return value;
}
}  // namespace

class MediaSessionState {
public:
    enum class Lifecycle {
        active,
        closing,
        closed,
        quarantined
    };

    struct Slot {
        OwnedMediaAllocation owned;
        void* pointer = nullptr;
    };

    std::shared_ptr<Resources> resources;
    std::shared_ptr<MediaSessionActivity> activity;
    std::unique_ptr<MediaDeviceOperations> operations;
    Workload workload;
    std::vector<MediaDeviceLimit> limits;
    const std::atomic_bool& cancel;
    MediaResidencyPlan plan;
    MediaAdmissionEvidence evidence;
    std::vector<Slot> slots;
    Footprint charge;
    Handle reservation = 0;
    bool admitted = false;
    Lifecycle lifecycle = Lifecycle::active;
    bool activity_owned = true;

    MediaSessionState(
        std::shared_ptr<Resources> r,
        std::shared_ptr<MediaSessionActivity> a,
        std::unique_ptr<MediaDeviceOperations> o,
        Workload w,
        std::vector<MediaDeviceLimit> l,
        const std::atomic_bool& c
    )
        : resources(std::move(r)), activity(std::move(a)), operations(std::move(o)),
          workload(w), limits(std::move(l)), cancel(c) {
        if (!resources || !activity || !operations || limits.empty())
            throw std::invalid_argument("media_session_owner");
        charge.resize(resources->snapshot().capacity.size());
        std::set<int> devices;
        for (const auto& limit : limits)
            if (limit.device < 0 || std::size_t(limit.device) + 1 >= charge.size()
                || !devices.insert(limit.device).second)
                throw std::invalid_argument("media_session_device");
        ++activity->owners;
    }

    ~MediaSessionState() {
        try {
            cleanup();
        } catch (...) {
        }
    }

    void healthy() {
        if (lifecycle != Lifecycle::active)
            throw std::runtime_error("media_session_closed_or_quarantined");
        if (cancel.load()) {
            const auto first =
                std::make_exception_ptr(std::runtime_error("media_session_cancelled"));
            rethrow_after_cleanup(first, [&] { cleanup(); });
        }
    }

    void drain() {
        std::exception_ptr first;
        // Visit every device even if the first drain fails or submission threw.
        for (const auto& limit : limits) {
            try {
                operations->synchronize(limit.device);
            } catch (...) {
                if (!first)
                    first = std::current_exception();
            }
        }
        if (first) {
            lifecycle = Lifecycle::quarantined;
            std::rethrow_exception(first);
        }
    }

    void cleanup() {
        if (lifecycle == Lifecycle::closed || lifecycle == Lifecycle::quarantined)
            return;
        lifecycle = Lifecycle::closing;
        drain();
        std::exception_ptr first;
        for (auto& slot : slots) {
            if (!slot.pointer)
                continue;
            try {
                operations->free(slot.owned.device, slot.pointer);
                slot.pointer = nullptr;
            } catch (...) {
                if (!first)
                    first = std::current_exception();
            }
        }
        if (first) {
            lifecycle = Lifecycle::quarantined;
            std::rethrow_exception(first);
        }
        if (reservation) {
            resources->released(reservation);
            reservation = 0;
        }
        lifecycle = Lifecycle::closed;
        if (activity_owned) {
            --activity->owners;
            activity_owned = false;
        }
    }

    void probe() {
        evidence = {};
        const auto snapshot = resources->snapshot();
        for (const auto& l : limits)
            evidence.budgets.push_back(
                {l.device,
                 snapshot.capacity[l.device + 1] - snapshot.used[l.device + 1],
                 operations->free_memory(l.device),
                 l.context_unaccounted,
                 l.safety_margin}
            );
        for (const auto& a : limits)
            for (const auto& b : limits)
                if (a.device != b.device)
                    evidence.peers.push_back(
                        {a.device,
                         b.device,
                         operations->direct_peer(a.device, b.device)}
                    );
    }
};

MediaSession::MediaSession(
    std::shared_ptr<Resources> r,
    std::shared_ptr<MediaSessionActivity> a,
    std::unique_ptr<MediaDeviceOperations> o,
    Workload w,
    std::vector<MediaDeviceLimit> limits,
    const std::atomic_bool& cancel
)
    : state_(
          std::make_shared<MediaSessionState>(
              std::move(r), std::move(a), std::move(o), w, std::move(limits), cancel
          )
      ) {
}

MediaSession::~MediaSession() {
    if (state_) {
        try {
            state_->cleanup();
        } catch (...) {
        }
    }
}

MediaSession& MediaSession::operator=(MediaSession&& other) noexcept {
    if (this != &other) {
        if (state_) {
            try {
                state_->cleanup();
            } catch (...) {
            }
        }
        state_ = std::move(other.state_);
    }
    return *this;
}

std::optional<MediaPlanError>
MediaSession::admit(const StageManifest& m, const MediaShape& shape) {
    auto& s = *state_;
    s.healthy();
    // This prefix admits one stage; drained owner reuse arrives with view lifetimes.
    if (s.admitted)
        return MediaPlanError{
            PlanFailure::unsupported_layout, "media_session_first_stage_only", {}
        };
    if (m.allocation_alignment > s.operations->allocation_alignment())
        return MediaPlanError{
            PlanFailure::unsupported_layout, "media_session_allocation_alignment", {}
        };
    s.probe();
    auto result = plan_media_stage(m, shape, s.evidence.budgets, s.evidence.peers);
    auto candidate = std::get_if<MediaResidencyPlan>(&result);
    if (!candidate)
        return std::get<MediaPlanError>(result);
    try {
        s.drain();
        s.plan = *candidate;
        std::size_t count = 0;
        for (const auto& d : s.plan.devices)
            count += d.allocations.size();
        s.slots.reserve(s.slots.size() + count);
        for (const auto& d : s.plan.devices)
            s.charge[d.device + 1] = d.new_ledger;
        s.reservation = s.resources->reserve(s.workload, s.charge);
        for (const auto& d : s.plan.devices) {
            for (const auto& allocation : d.allocations) {
                const auto bytes = aligned(allocation.bytes, m.allocation_alignment);
                auto owned = allocation;
                owned.bytes = bytes;
                MediaSessionState::Slot slot;
                slot.owned = {
                    "media/" + std::to_string(owner_generation()),
                    d.device,
                    m.stage,
                    owned,
                    bytes,
                    m.allocation_alignment
                };
                s.slots.push_back(std::move(slot));
                s.slots.back().pointer =
                    s.operations->allocate(d.device, bytes, m.allocation_alignment);
                if (!s.slots.back().pointer)
                    throw std::runtime_error("media_session_allocation_failed");
            }
        }
        s.admitted = true;
        return {};
    } catch (...) {
        abort(std::current_exception());
        throw;
    }
}

const MediaResidencyPlan& MediaSession::plan() const {
    return state_->plan;
}

const MediaAdmissionEvidence& MediaSession::evidence() const {
    return state_->evidence;
}

void MediaSession::finish() {
    if (state_->lifecycle == MediaSessionState::Lifecycle::quarantined)
        throw std::runtime_error("media_session_cleanup_unconfirmed");
    if (state_->cancel.load()) {
        abort(std::make_exception_ptr(std::runtime_error("media_session_cancelled")));
    }
    state_->cleanup();
}

void MediaSession::abort(std::exception_ptr first) {
    if (state_->lifecycle == MediaSessionState::Lifecycle::quarantined) {
        if (first)
            std::rethrow_exception(first);
        throw std::runtime_error("media_session_cleanup_unconfirmed");
    }
    try {
        state_->cleanup();
    } catch (...) {
        if (first)
            throw OperationCleanupError(first, std::current_exception());
        throw;
    }
    if (first)
        std::rethrow_exception(first);
}
}  // namespace kadan
