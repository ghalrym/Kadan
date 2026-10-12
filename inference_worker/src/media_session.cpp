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

    struct InitializedRange {
        Bytes begin = 0, end = 0;
        auto operator<=>(const InitializedRange&) const = default;
    };

    struct Slot {
        struct GatherProgress {
            InitializedRange region;
            std::size_t next_column;
        };

        OwnedMediaAllocation owned;
        void* pointer = nullptr;
        std::uint64_t generation = 0;
        std::size_t views = 0;
        MediaEvent ready;
        std::vector<MediaEvent> consumers;
        std::string logical_name;
        DeviceDType dtype = DeviceDType::u8;
        std::vector<std::size_t> dimensions, strides;
        std::vector<InitializedRange> initialized;
        std::optional<GatherProgress> gather;

        bool contains(Bytes offset, Bytes bytes) const {
            return std::any_of(
                initialized.begin(), initialized.end(), [&](const auto& r) {
                    return r.begin <= offset && offset + bytes <= r.end;
                }
            );
        }

        void certify(Bytes offset, Bytes bytes) {
            initialized.emplace_back(offset, add(offset, bytes));
            std::sort(initialized.begin(), initialized.end());
            std::size_t count = 0;
            for (const auto& range : initialized) {
                if (count && range.begin <= initialized[count - 1].end)
                    initialized[count - 1].end =
                        std::max(initialized[count - 1].end, range.end);
                else
                    initialized[count++] = range;
            }
            initialized.resize(count);
        }

        void invalidate(Bytes offset, Bytes bytes) {
            gather.reset();
            std::vector<InitializedRange> retained;
            for (const auto& range : initialized) {
                if (range.end <= offset || range.begin >= offset + bytes)
                    retained.push_back(range);
                else {
                    if (range.begin < offset)
                        retained.emplace_back(range.begin, offset);
                    if (range.end > offset + bytes)
                        retained.emplace_back(offset + bytes, range.end);
                }
            }
            initialized.swap(retained);
        }
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
    MediaTransferCounters transfers;
    std::vector<Slot> slots;
    std::map<std::pair<int, std::size_t>, std::size_t> mapping;
    Footprint charge;
    Handle reservation = 0;
    std::set<MediaEvent> unchecked;
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
        for (auto& slot : slots)
            slot.consumers.clear();
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
        for (const auto& limit : limits) {
            try {
                operations->release_events(limit.device);
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

    void dependencies(int device, Slot& slot) {
        if (slot.ready.id)
            operations->wait(device, slot.ready);
        for (const auto& event : slot.consumers)
            operations->wait(device, event);
    }
};

MediaSessionState::Slot& MediaSessionState::view(DeviceTensorView& v) {
    if (v.state_.get() != this || v.slot_ >= slots.size()
        || v.layout_.generation != slots[v.slot_].generation || !slots[v.slot_].pointer)
        throw std::invalid_argument("media_session_foreign_view");
    return slots[v.slot_];
}

DeviceTensorView::DeviceTensorView(DeviceTensorView&& v) noexcept
    : state_(std::move(v.state_)), slot_(v.slot_), layout_(std::move(v.layout_)),
      name_(std::move(v.name_)), produced_(v.produced_) {
}

DeviceTensorView& DeviceTensorView::operator=(DeviceTensorView&& v) noexcept {
    if (this != &v) {
        release();
        state_ = std::move(v.state_);
        slot_ = v.slot_;
        layout_ = std::move(v.layout_);
        name_ = std::move(v.name_);
        produced_ = v.produced_;
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

MediaEvent DeviceTensorView::readiness() const {
    if (!state_)
        throw std::runtime_error("media_session_empty_view");
    return state_->slots[slot_].ready;
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
    if (!s.unchecked.empty())
        throw std::runtime_error("media_session_unchecked_status");
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
        for (const auto& limit : s.limits)
            s.operations->release_events(limit.device);
        for (auto& slot : s.slots) {
            slot.ready = {slot.owned.device, 0};
            slot.consumers.clear();
            slot.gather.reset();
        }
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
        for (auto& slot : s.slots)
            if (!slot.owned.allocation.weight && slot.logical_name != "resident/hidden"
                && !std::any_of(m.tensors.begin(), m.tensors.end(), [&](const auto& t) {
                       return t.name == slot.logical_name
                              && t.first == MediaPhase::prepare
                              && t.last == MediaPhase::complete;
                   }))
                slot.initialized.clear();
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

const MediaTransferCounters& MediaSession::transfers() const {
    return state_->transfers;
}

void MediaSession::advance(MediaPhase phase) {
    auto& s = *state_;
    s.healthy();
    const bool next_block = s.phase == MediaPhase::ff && phase == MediaPhase::kv;
    if (!s.unchecked.empty())
        throw std::runtime_error("media_session_unchecked_status");
    if (!s.admitted || (phase < s.phase && !next_block) || phase > MediaPhase::complete)
        throw std::invalid_argument("media_session_phase");
    for (const auto& live : s.plan.tensors)
        if (((live.last < phase && live.last >= s.phase)
             || (next_block && live.first != MediaPhase::prepare))
            && s.slots[s.mapping.at({live.device, live.allocation})].views)
            throw std::runtime_error("media_session_live_alias");
    try {
        s.drain();
        if (s.phase <= MediaPhase::kv && phase >= MediaPhase::attention
            && !s.plan.hidden_write_phases.empty()) {
            for (const auto& live : s.plan.tensors)
                if (live.name == "resident/key" || live.name == "resident/value") {
                    const auto& slot =
                        s.slots[s.mapping.at({live.device, live.allocation})];
                    if (slot.logical_name != live.name || !slot.contains(0, live.bytes))
                        throw std::runtime_error("media_session_incomplete_kv");
                }
        }
        if (next_block) {
            for (auto& slot : s.slots)
                if (slot.logical_name != "resident/hidden"
                    && !slot.owned.allocation.weight
                    && !std::any_of(
                        s.manifest.tensors.begin(),
                        s.manifest.tensors.end(),
                        [&](const auto& t) {
                            return t.name == slot.logical_name
                                   && t.first == MediaPhase::prepare
                                   && t.last == MediaPhase::complete;
                        }
                    ))
                    slot.initialized.clear();
        }
        for (const auto& limit : s.limits)
            s.operations->release_events(limit.device);
        for (auto& slot : s.slots) {
            slot.ready = {slot.owned.device, 0};
            slot.gather.reset();
        }
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
        slot.initialized.clear();
        slot.gather.reset();
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

void MediaSession::upload(
    DeviceTensorView& v, std::span<const std::byte> bytes, bool weight
) {
    auto& s = *state_;
    s.healthy();
    auto& slot = s.view(v);
    if (s.phase != MediaPhase::prepare || v.layout_.offset
        || (slot.owned.allocation.kind == AllocationKind::weight) != weight
        || (!weight && slot.owned.allocation.kind != AllocationKind::activation)
        || bytes.size() != v.layout_.bytes)
        throw std::invalid_argument("media_session_upload_boundary");
    try {
        s.dependencies(v.layout_.device, slot);
        slot.gather.reset();
        s.operations->upload(v.layout_.device, slot.pointer, bytes);
        slot.ready = s.operations->record(v.layout_.device);
        s.operations->synchronize(v.layout_.device);
        slot.certify(v.layout_.offset, v.layout_.bytes);
        auto& counter = weight ? s.transfers.weight_upload : s.transfers.input_upload;
        counter = add(counter, bytes.size());
    } catch (...) {
        abort(std::current_exception());
    }
}

void MediaSession::upload_input(DeviceTensorView& v, std::span<const std::byte> bytes) {
    upload(v, bytes, false);
}

void MediaSession::upload_weight(
    DeviceTensorView& v, std::span<const std::byte> bytes
) {
    upload(v, bytes, true);
}

void MediaSession::download_output(DeviceTensorView& v, std::span<std::byte> bytes) {
    auto& s = *state_;
    s.healthy();
    auto& slot = s.view(v);
    if (s.phase != MediaPhase::complete
        || !slot.contains(v.layout_.offset, v.layout_.bytes)
        || bytes.size() != v.layout_.bytes
        || slot.owned.allocation.kind != AllocationKind::activation)
        throw std::invalid_argument("media_session_output_boundary");
    try {
        s.dependencies(v.layout_.device, slot);
        s.operations->download(v.layout_.device, bytes, v.layout_.data);
        s.operations->synchronize(v.layout_.device);
        s.transfers.output_download = add(s.transfers.output_download, bytes.size());
    } catch (...) {
        abort(std::current_exception());
    }
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

void MediaSession::gather(
    DeviceTensorView& source, DeviceTensorView& destination, std::size_t first_column
) {
    auto& s = *state_;
    s.healthy();
    auto& a = s.view(source);
    auto& b = s.view(destination);
    const auto& x = source.layout_;
    const auto& y = destination.layout_;
    if (&a == &b || a.owned.allocation.weight || !a.contains(x.offset, x.bytes)
        || x.dtype != y.dtype || x.dimensions.size() != 2 || y.dimensions.size() != 2
        || x.dimensions[0] != y.dimensions[0]
        || x.strides != std::vector<std::size_t>{x.dimensions[1], 1}
        || y.strides != std::vector<std::size_t>{y.dimensions[1], 1}
        || first_column > y.dimensions[1]
        || x.dimensions[1] > y.dimensions[1] - first_column || b.owned.allocation.weight
        || ((destination.name_ == "resident/key"
             || destination.name_ == "resident/value")
            && s.phase != MediaPhase::kv)
        || (destination.name_ == "resident/hidden"
            && s.phase <= s.plan.kv_complete_before_hidden_write))
        throw std::invalid_argument("media_session_gather_layout");
    if (x.device != y.device
        && !std::any_of(
            s.evidence.peers.begin(), s.evidence.peers.end(), [&](const auto& p) {
                return p.from == x.device && p.to == y.device && p.direct;
            }
        ))
        throw std::runtime_error("media_session_direct_peer_required");
    const auto bytes = mul(x.dimensions[1], width(x.dtype));
    if (s.unchecked.contains(b.ready))
        throw std::invalid_argument("media_session_unchecked_overwrite");
    const MediaSessionState::InitializedRange region{y.offset, add(y.offset, y.bytes)};
    if (first_column
        && (!b.gather || b.gather->region != region
            || b.gather->next_column != first_column))
        throw std::invalid_argument("media_session_gather_cycle");
    try {
        s.dependencies(y.device, a);
        s.dependencies(y.device, b);
        a.consumers.reserve(a.consumers.size() + 1);
        if (!first_column)
            b.invalidate(y.offset, y.bytes);
        for (std::size_t row = 0; row < x.dimensions[0]; ++row) {
            const auto offset =
                mul(add(mul(row, y.dimensions[1]), first_column), width(y.dtype));
            b.invalidate(add(y.offset, offset), bytes);
            s.operations->peer(
                x.device,
                static_cast<std::byte*>(x.data) + row * bytes,
                y.device,
                static_cast<std::byte*>(y.data) + offset,
                bytes
            );
        }
        b.ready = s.operations->record(y.device);
        a.consumers.push_back(b.ready);
        // A peer copy preserves a validated source. Coverage becomes usable only
        // after the destination event completes, including every contributing shard.
        s.operations->synchronize(y.device);
        for (std::size_t row = 0; row < x.dimensions[0]; ++row)
            b.certify(
                add(y.offset,
                    mul(add(mul(row, y.dimensions[1]), first_column), width(y.dtype))),
                bytes
            );
        const auto next_column = first_column + x.dimensions[1];
        if (next_column < y.dimensions[1])
            b.gather = MediaSessionState::Slot::GatherProgress{region, next_column};
        if (x.device != y.device)
            s.transfers.peer_activation = add(s.transfers.peer_activation, x.bytes);
        else
            s.transfers.local_activation = add(s.transfers.local_activation, x.bytes);
    } catch (...) {
        abort(std::current_exception());
    }
}

void MediaSession::launch(
    int device,
    std::span<DeviceTensorView* const> inputs,
    std::span<DeviceTensorView* const> outputs,
    const Launch& launch
) {
    auto& s = *state_;
    s.healthy();
    if (outputs.empty() || !launch)
        throw std::invalid_argument("media_session_launch");
    std::vector<MediaDeviceSpan> read, write;
    std::set<std::size_t> output_slots;
    for (auto* v : inputs) {
        if (!v)
            throw std::invalid_argument("media_session_null_view");
        auto& slot = s.view(*v);
        if (v->layout_.device != device
            || !slot.contains(v->layout_.offset, v->layout_.bytes))
            throw std::invalid_argument("media_session_unready_input");
        read.push_back(v->layout_);
    }
    for (auto* v : outputs) {
        if (!v)
            throw std::invalid_argument("media_session_null_view");
        auto& slot = s.view(*v);
        if (v->layout_.device != device || slot.owned.allocation.weight
            || !output_slots.insert(v->slot_).second
            || ((v->name_ == "resident/key" || v->name_ == "resident/value")
                && s.phase != MediaPhase::kv)
            || (v->name_ == "resident/hidden"
                && s.phase <= s.plan.kv_complete_before_hidden_write))
            throw std::invalid_argument("media_session_write_ownership");
        if (s.unchecked.contains(slot.ready))
            throw std::invalid_argument("media_session_unchecked_overwrite");
        write.push_back(v->layout_);
    }
    try {
        for (auto* v : inputs) {
            auto& slot = s.view(*v);
            s.dependencies(device, slot);
            slot.consumers.reserve(slot.consumers.size() + 1);
        }
        for (auto* v : outputs) {
            s.dependencies(device, s.view(*v));
            s.view(*v).invalidate(v->layout_.offset, v->layout_.bytes);
        }
        launch(s.operations->stream(device), read, write);
        const auto event = s.operations->record(device);
        s.unchecked.insert(event);
        for (auto* v : inputs)
            s.view(*v).consumers.push_back(event);
        for (auto* v : outputs) {
            s.view(*v).ready = event;
            v->produced_ = event;
        }
    } catch (...) {
        abort(std::current_exception());
    }
}

void MediaSession::complete(
    std::span<DeviceTensorView* const> outputs, DeviceTensorView& status
) {
    auto& s = *state_;
    s.healthy();
    auto& flag = s.view(status);
    if (outputs.empty() || status.layout_.dtype != DeviceDType::u32
        || status.layout_.bytes != 4
        || flag.owned.allocation.kind != AllocationKind::status || !flag.ready.id
        || status.produced_ != flag.ready)
        throw std::invalid_argument("media_session_status");
    for (auto* v : outputs)
        if (!v || s.view(*v).ready != flag.ready || v->produced_ != status.produced_)
            throw std::invalid_argument("media_session_status_event");
    try {
        s.operations->synchronize(flag.ready.device);
        const auto invalid = s.operations->read_status(flag.ready.device, flag.pointer);
        s.transfers.control_download = add(s.transfers.control_download, 4);
        if (invalid)
            throw std::runtime_error("media_session_nonfinite_output");
        for (auto* v : outputs)
            s.view(*v).certify(v->layout_.offset, v->layout_.bytes);
        flag.certify(status.layout_.offset, status.layout_.bytes);
        s.unchecked.erase(flag.ready);
    } catch (...) {
        abort(std::current_exception());
    }
}

void MediaSession::finish() {
    if (state_->lifecycle == MediaSessionState::Lifecycle::quarantined)
        throw std::runtime_error("media_session_cleanup_unconfirmed");
    if (!state_->unchecked.empty())
        abort(
            std::make_exception_ptr(
                std::runtime_error("media_session_unchecked_status")
            )
        );
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
