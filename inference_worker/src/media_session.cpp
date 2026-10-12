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

Bytes width(DeviceDType type) {
    switch (type) {
    case DeviceDType::bf16:
    case DeviceDType::f16:
        return 2;
    case DeviceDType::f32:
    case DeviceDType::u32:
        return 4;
    case DeviceDType::f64:
        return 8;
    case DeviceDType::i8:
    case DeviceDType::u8:
        return 1;
    }
    throw std::invalid_argument("media_session_dtype");
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
        std::uint64_t generation = 0;
        std::size_t views = 0;
        std::string logical_name;
        DeviceDType dtype = DeviceDType::u8;
        std::vector<std::size_t> dimensions, strides;
    };

    std::shared_ptr<Resources> resources;
    std::shared_ptr<MediaSessionActivity> activity;
    std::unique_ptr<MediaDeviceOperations> operations;
    Workload workload;
    std::vector<MediaDeviceLimit> limits;
    const std::atomic_bool& cancel;
    MediaResidencyPlan plan;
    StageManifest manifest;
    MediaShape shape;
    MediaAdmissionEvidence evidence;
    std::vector<Slot> slots;
    std::map<std::pair<int, std::size_t>, std::size_t> mapping;
    Footprint charge;
    Handle reservation = 0;
    MediaPhase phase = MediaPhase::prepare;
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

    void no_views() const {
        for (const auto& slot : slots)
            if (slot.views)
                throw std::runtime_error("media_session_live_view");
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
        // Views own the physical allocation after a request error/destruction.
        // Last-view release retries confirmed cleanup; admission stays closed.
        for (const auto& slot : slots)
            if (slot.views)
                return;
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

    std::vector<OwnedMediaAllocation> owned() const {
        std::vector<OwnedMediaAllocation> result;
        for (const auto& slot : slots)
            if (slot.pointer)
                result.push_back(slot.owned);
        return result;
    }

    Slot& view(DeviceTensorView& v);
};

MediaSessionState::Slot& MediaSessionState::view(DeviceTensorView& v) {
    if (v.state_.get() != this || v.slot_ >= slots.size()
        || v.layout_.generation != slots[v.slot_].generation || !slots[v.slot_].pointer)
        throw std::invalid_argument("media_session_foreign_view");
    return slots[v.slot_];
}

DeviceTensorView::DeviceTensorView(DeviceTensorView&& v) noexcept
    : state_(std::move(v.state_)), slot_(v.slot_), layout_(std::move(v.layout_)),
      name_(std::move(v.name_)) {
}

DeviceTensorView& DeviceTensorView::operator=(DeviceTensorView&& v) noexcept {
    if (this != &v) {
        release();
        state_ = std::move(v.state_);
        slot_ = v.slot_;
        layout_ = std::move(v.layout_);
        name_ = std::move(v.name_);
    }
    return *this;
}

DeviceTensorView::~DeviceTensorView() {
    release();
}

void DeviceTensorView::release() noexcept {
    if (state_) {
        --state_->slots[slot_].views;
        if (state_->lifecycle == MediaSessionState::Lifecycle::closing) {
            try {
                state_->cleanup();
            } catch (...) {
            }
        }
        state_.reset();
    }
}

const MediaDeviceSpan& DeviceTensorView::layout() const {
    if (!state_)
        throw std::runtime_error("media_session_empty_view");
    return layout_;
}

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
    s.no_views();
    if (s.admitted && s.phase != MediaPhase::complete)
        throw std::runtime_error("media_session_stage_boundary");
    if (m.allocation_alignment > s.operations->allocation_alignment())
        return MediaPlanError{
            PlanFailure::unsupported_layout, "media_session_allocation_alignment", {}
        };
    s.probe();
    auto result =
        plan_media_stage(m, shape, s.evidence.budgets, s.evidence.peers, s.owned());
    auto candidate = std::get_if<MediaResidencyPlan>(&result);
    if (!candidate) {
        const auto& error = std::get<MediaPlanError>(result);
        if (error.reason != PlanFailure::capacity || !error.rejected)
            return error;
        candidate = &*std::get<MediaPlanError>(result).rejected;
    }
    try {
        s.drain();
        // Reuse exactly once. Unmatched dormant activations and obsolete weight
        // geometry are dead at this drained, view-free stage boundary.
        std::set<std::string> reuse;
        for (const auto& r : candidate->reuse)
            if (!reuse.insert(r.owner).second)
                throw std::runtime_error("media_session_duplicate_reuse");
        const std::set<std::string> retirement(
            candidate->retire_owners.begin(), candidate->retire_owners.end()
        );
        for (const auto& owner : retirement)
            if (reuse.contains(owner))
                throw std::runtime_error("media_session_retire_reused_owner");
        for (auto& slot : s.slots) {
            if (!slot.pointer || reuse.contains(slot.owned.owner))
                continue;
            try {
                s.operations->free(slot.owned.device, slot.pointer);
            } catch (...) {
                s.lifecycle = MediaSessionState::Lifecycle::quarantined;
                throw;
            }
            slot.pointer = nullptr;
        }
        std::erase_if(s.slots, [](const auto& slot) { return !slot.pointer; });
        std::fill(s.charge.begin(), s.charge.end(), 0);
        for (const auto& slot : s.slots)
            if (slot.pointer)
                s.charge[slot.owned.device + 1] =
                    add(s.charge[slot.owned.device + 1], slot.owned.ledger_bytes);
        if (s.reservation)
            s.resources->resize_loading(s.reservation, s.charge);
        s.admitted = false;
        // Never allocate on the retirement credit: confirmed frees, fresh device
        // memory and ledger evidence determine the final plan.
        s.probe();
        result =
            plan_media_stage(m, shape, s.evidence.budgets, s.evidence.peers, s.owned());
        candidate = std::get_if<MediaResidencyPlan>(&result);
        if (!candidate)
            return std::get<MediaPlanError>(result);
        s.plan = *candidate;
        s.manifest = m;
        s.shape = shape;
        s.mapping.clear();
        std::size_t count = 0;
        for (const auto& d : s.plan.devices)
            count += d.allocations.size();
        s.slots.reserve(s.slots.size() + count);
        auto charge = s.charge;
        for (const auto& d : s.plan.devices)
            charge[d.device + 1] = add(charge[d.device + 1], d.new_ledger);
        if (s.reservation)
            s.resources->resize_loading(s.reservation, charge);
        else
            s.reservation = s.resources->reserve(s.workload, charge);
        s.charge = std::move(charge);
        for (const auto& d : s.plan.devices) {
            for (std::size_t index = 0; index < d.allocations.size(); ++index) {
                const auto reused = std::find_if(
                    s.plan.reuse.begin(), s.plan.reuse.end(), [&](const auto& r) {
                        return r.device == d.device && r.allocation == index;
                    }
                );
                std::size_t slot_index;
                if (reused != s.plan.reuse.end()) {
                    const auto found = std::find_if(
                        s.slots.begin(), s.slots.end(), [&](const auto& slot) {
                            return slot.pointer && slot.owned.owner == reused->owner;
                        }
                    );
                    if (found == s.slots.end())
                        throw std::runtime_error("media_session_missing_owner");
                    slot_index = std::size_t(found - s.slots.begin());
                    found->owned.stage = m.stage;
                } else {
                    const auto bytes =
                        aligned(d.allocations[index].bytes, m.allocation_alignment);
                    const auto generation = owner_generation();
                    auto allocation = d.allocations[index];
                    allocation.bytes = bytes;
                    MediaSessionState::Slot slot;
                    slot.owned = {
                        "media/" + std::to_string(generation),
                        d.device,
                        m.stage,
                        allocation,
                        bytes,
                        m.allocation_alignment
                    };
                    slot.generation = generation;
                    s.slots.push_back(std::move(slot));
                    slot_index = s.slots.size() - 1;
                    s.slots.back().pointer =
                        s.operations->allocate(d.device, bytes, m.allocation_alignment);
                    if (!s.slots.back().pointer)
                        throw std::runtime_error("media_session_allocation_failed");
                }
                s.mapping.emplace(std::pair{d.device, index}, slot_index);
            }
        }
        s.phase = MediaPhase::prepare;
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

void MediaSession::advance(MediaPhase phase) {
    auto& s = *state_;
    s.healthy();
    const bool next_block = s.phase == MediaPhase::ff && phase == MediaPhase::kv;
    if (!s.admitted || (phase < s.phase && !next_block) || phase > MediaPhase::complete)
        throw std::invalid_argument("media_session_phase");
    for (const auto& live : s.plan.tensors)
        if (((live.last < phase && live.last >= s.phase)
             || (next_block && live.first != MediaPhase::prepare))
            && s.slots[s.mapping.at({live.device, live.allocation})].views)
            throw std::runtime_error("media_session_live_alias");
    try {
        s.drain();
        s.phase = phase;
    } catch (...) {
        abort(std::current_exception());
    }
}

DeviceTensorView MediaSession::tensor(
    int device,
    std::string_view name,
    DeviceDType type,
    std::vector<std::size_t> dimensions,
    std::vector<std::size_t> strides
) {
    auto& s = *state_;
    s.healthy();
    if (!s.admitted)
        throw std::runtime_error("media_session_not_admitted");
    std::size_t allocation = 0;
    Bytes limit = 0;
    auto live =
        std::find_if(s.plan.tensors.begin(), s.plan.tensors.end(), [&](const auto& t) {
            return t.device == device && t.name == name;
        });
    if (live != s.plan.tensors.end()) {
        if (s.phase < live->first || s.phase > live->last)
            throw std::runtime_error("media_session_tensor_phase");
        allocation = live->allocation;
        limit = live->bytes;
    } else {
        const auto d = std::find_if(
            s.plan.devices.begin(), s.plan.devices.end(), [&](const auto& p) {
                return p.device == device;
            }
        );
        if (d == s.plan.devices.end())
            throw std::invalid_argument("media_session_tensor_device");
        const auto a = std::find_if(
            d->allocations.begin(), d->allocations.end(), [&](const auto& p) {
                return p.weight && p.name == name;
            }
        );
        if (a == d->allocations.end())
            throw std::invalid_argument("media_session_tensor_name");
        allocation = std::size_t(a - d->allocations.begin());
        limit = a->bytes;
    }
    if (name == "resident/hidden" || name == "resident/key"
        || name == "resident/value") {
        const auto d = std::find_if(
            s.plan.devices.begin(), s.plan.devices.end(), [&](const auto& p) {
                return p.device == device;
            }
        );
        const auto columns =
            name == "resident/hidden"
                ? s.shape.hidden
                : d->heads / (s.shape.heads / s.shape.kv_heads) * s.shape.head_dim;
        if (type != DeviceDType::f32
            || dimensions != std::vector<std::size_t>{s.shape.tokens, columns}
            || strides != std::vector<std::size_t>{columns, 1})
            throw std::invalid_argument("media_session_persistent_layout");
    }
    if (dimensions.empty() || dimensions.size() != strides.size())
        throw std::invalid_argument("media_session_dimensions");
    Bytes elements = 1, end = 1;
    for (std::size_t i = dimensions.size(); i-- > 0;) {
        if (!dimensions[i] || !strides[i] || strides[i] < elements)
            throw std::invalid_argument("media_session_strides");
        end = add(end, mul(dimensions[i] - 1, strides[i]));
        elements = mul(dimensions[i], strides[i]);
    }
    const auto bytes = mul(end, width(type));
    if (bytes > limit)
        throw std::invalid_argument("media_session_tensor_extent");
    const auto index = s.mapping.at({device, allocation});
    auto& slot = s.slots[index];
    if (slot.views)
        throw std::runtime_error("media_session_slot_in_use");
    if (slot.owned.allocation.weight) {
        const auto weight = std::find_if(
            s.manifest.weights.begin(), s.manifest.weights.end(), [&](const auto& w) {
                return slot.owned.allocation.weight == w.identity;
            }
        );
        if (weight == s.manifest.weights.end())
            throw std::invalid_argument("media_session_weight_manifest");
        const auto expected =
            weight->storage == TensorStorage::bf16_exact   ? DeviceDType::bf16
            : weight->storage == TensorStorage::f16_exact  ? DeviceDType::f16
            : weight->storage == TensorStorage::f32        ? DeviceDType::f32
            : weight->storage == TensorStorage::i8_convrot ? DeviceDType::i8
                                                           : DeviceDType::u8;
        if (type != expected
            || dimensions
                   != std::vector<
                       std::size_t>{slot.owned.allocation.rows, weight->columns}
            || strides != std::vector<std::size_t>{weight->columns, 1})
            throw std::invalid_argument("media_session_weight_layout");
    }
    if (slot.logical_name != name || slot.dtype != type || slot.dimensions != dimensions
        || slot.strides != strides) {
        slot.logical_name = name;
        slot.dtype = type;
        slot.dimensions = dimensions;
        slot.strides = strides;
    }
    DeviceTensorView result;
    result.layout_ = {
        slot.pointer,
        device,
        type,
        std::move(dimensions),
        std::move(strides),
        bytes,
        slot.generation
    };
    result.name_ = name;
    result.slot_ = index;
    result.state_ = state_;
    ++slot.views;
    return result;
}

DeviceTensorView MediaSession::weight(const WeightSlice& slice) {
    auto& s = *state_;
    s.healthy();
    if (!std::any_of(
            s.plan.weights.begin(), s.plan.weights.end(), [&](const auto& planned) {
                return planned.name == slice.name && planned.identity == slice.identity
                       && planned.device == slice.device
                       && planned.first_row == slice.first_row
                       && planned.rows == slice.rows
                       && planned.tile_columns == slice.tile_columns
                       && planned.stored_bytes == slice.stored_bytes;
            }
        ))
        throw std::invalid_argument("media_session_weight_slice");
    const auto d =
        std::find_if(s.plan.devices.begin(), s.plan.devices.end(), [&](const auto& p) {
            return p.device == slice.device;
        });
    if (d == s.plan.devices.end())
        throw std::invalid_argument("media_session_weight_device");
    const auto a =
        std::find_if(d->allocations.begin(), d->allocations.end(), [&](const auto& p) {
            return p.weight == slice.identity && p.first_row == slice.first_row
                   && p.rows == slice.rows && p.bytes == slice.stored_bytes
                   && p.name.starts_with("weight/" + slice.name + "/");
        });
    if (a == d->allocations.end())
        throw std::invalid_argument("media_session_weight_slice");
    const auto w = std::find_if(
        s.manifest.weights.begin(), s.manifest.weights.end(), [&](const auto& p) {
            return p.name == slice.name && p.identity == slice.identity;
        }
    );
    if (w == s.manifest.weights.end() || !slice.tile_columns
        || slice.tile_columns > slice.rows)
        throw std::invalid_argument("media_session_weight_manifest");
    const auto type = w->storage == TensorStorage::bf16_exact   ? DeviceDType::bf16
                      : w->storage == TensorStorage::f16_exact  ? DeviceDType::f16
                      : w->storage == TensorStorage::f32        ? DeviceDType::f32
                      : w->storage == TensorStorage::i8_convrot ? DeviceDType::i8
                                                                : DeviceDType::u8;
    return tensor(
        slice.device, a->name, type, {slice.rows, w->columns}, {w->columns, 1}
    );
}

DeviceTensorView
MediaSession::rows(DeviceTensorView& parent, std::size_t first, std::size_t count) {
    auto& s = *state_;
    s.healthy();
    s.view(parent);
    const auto& p = parent.layout_;
    if (p.dimensions.size() != 2
        || p.strides != std::vector<std::size_t>{p.dimensions[1], 1} || !count
        || first > p.dimensions[0] || count > p.dimensions[0] - first)
        throw std::invalid_argument("media_session_row_region");
    DeviceTensorView result;
    result.layout_ = p;
    const auto offset = mul(mul(first, p.strides[0]), width(p.dtype));
    result.layout_.data = static_cast<std::byte*>(p.data) + offset;
    result.layout_.offset = add(p.offset, offset);
    result.layout_.dimensions[0] = count;
    result.layout_.bytes = mul(mul(count, p.strides[0]), width(p.dtype));
    result.name_ = parent.name_;
    result.slot_ = parent.slot_;
    result.state_ = state_;
    ++s.slots[parent.slot_].views;
    return result;
}

void MediaSession::finish() {
    if (state_->lifecycle == MediaSessionState::Lifecycle::quarantined)
        throw std::runtime_error("media_session_cleanup_unconfirmed");
    if (state_->cancel.load()) {
        abort(std::make_exception_ptr(std::runtime_error("media_session_cancelled")));
    }
    state_->cleanup();
    state_->no_views();
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
