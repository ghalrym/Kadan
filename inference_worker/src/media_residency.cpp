#include "kadan/media_residency.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>

#include "kadan/device_placement.hpp"

namespace kadan {
namespace {
Bytes add(Bytes a, Bytes b) {
    if (b > std::numeric_limits<Bytes>::max() - a)
        throw std::overflow_error("media_bytes_overflow");
    return a + b;
}

Bytes mul(Bytes a, Bytes b) {
    if (a && b > std::numeric_limits<Bytes>::max() / a)
        throw std::overflow_error("media_bytes_overflow");
    return a * b;
}

Bytes aligned(Bytes bytes, std::size_t alignment) {
    const auto tail = bytes % alignment;
    return tail ? add(bytes, alignment - tail) : bytes;
}

Bytes workspace_bytes(std::initializer_list<Bytes> parts, std::size_t alignment) {
    Bytes bytes = 0;
    for (auto part : parts)
        bytes = add(bytes, aligned(part, alignment));
    return bytes;
}

Bytes width(TensorStorage storage) {
    switch (storage) {
    case TensorStorage::bf16_exact:
    case TensorStorage::f16_exact:
        return 2;
    case TensorStorage::f32:
        return 4;
    case TensorStorage::i8_convrot:
    case TensorStorage::u8:
        return 1;
    }
    throw std::invalid_argument("media_storage");
}

bool transformer(MediaStage stage) {
    return stage == MediaStage::text || stage == MediaStage::denoise;
}

bool phase_valid(MediaPhase phase) {
    return phase >= MediaPhase::prepare && phase <= MediaPhase::complete;
}

void validate(
    const StageManifest& m,
    const MediaShape& s,
    std::span<const DeviceBudget> budgets,
    const PeerMatrix& peers,
    std::span<const OwnedMediaAllocation> owned
) {
    if (m.stage < MediaStage::text || m.stage > MediaStage::audio_decode || !s.tokens
        || !s.hidden || s.text_tokens > s.tokens || s.tokens > 2147483647
        || s.hidden > 2147483647 || budgets.empty() || !m.allocation_alignment
        || (m.allocation_alignment & (m.allocation_alignment - 1)))
        throw std::invalid_argument("media_shape_or_alignment");
    if (transformer(m.stage)
        && (!s.heads || !s.kv_heads || s.heads % s.kv_heads || !s.head_dim || !s.ff
            || s.head_dim > 2048 || s.ff > 2147483647 || s.attention_inner > 2147483647
            || mul(s.heads, s.head_dim) != s.attention_inner))
        throw std::invalid_argument("media_attention_shape");
    std::set<int> devices;
    for (const auto& b : budgets)
        if (b.device < 0 || !devices.insert(b.device).second)
            throw std::invalid_argument("media_devices");
    std::set<std::pair<int, int>> edges;
    for (const auto& p : peers)
        if (!devices.contains(p.from) || !devices.contains(p.to)
            || !edges.emplace(p.from, p.to).second)
            throw std::invalid_argument("media_peer_matrix");
    std::set<std::string> names;
    std::set<WeightIdentity> identities;
    for (const auto& w : m.weights) {
        if (w.name.empty() || !names.insert("weight/" + w.name).second
            || !identities.insert(w.identity).second || !w.rows || !w.columns
            || !w.planes || w.rows % w.planes || !w.alignment || w.rows > 2147483647
            || w.columns > 2147483647 || w.rows / w.planes % w.alignment
            || w.identity.payload_end <= w.identity.payload_begin
            || w.identity.payload_end - w.identity.payload_begin
                   != mul(mul(w.rows, w.columns), width(w.storage)))
            throw std::invalid_argument("media_weight_descriptor");
        if (w.partition < WeightPartition::replicated
            || w.partition > WeightPartition::kv_heads
            || w.precision < ProjectionPrecision::bf16_f32
            || w.precision > ProjectionPrecision::convrot)
            throw std::invalid_argument("media_weight_mode");
        if ((w.partition == WeightPartition::heads
             || w.partition == WeightPartition::kv_heads)
            && (!transformer(m.stage)
                || w.rows / w.planes
                       != mul(
                           w.partition == WeightPartition::kv_heads ? s.kv_heads
                                                                    : s.heads,
                           s.head_dim
                       )
                || s.head_dim % w.alignment))
            throw std::invalid_argument("media_weight_heads");
    }
    for (const auto& t : m.tensors)
        if (t.name.empty() || t.name.starts_with("resident/")
            || t.name.starts_with("weight/") || !names.insert(t.name).second || !t.bytes
            || !phase_valid(t.first) || !phase_valid(t.last) || t.first > t.last
            || t.kind <= AllocationKind::weight || t.kind > AllocationKind::status)
            throw std::invalid_argument("media_tensor_descriptor");
    std::set<std::string> owners;
    for (const auto& o : owned) {
        const auto& a = o.allocation;
        if (!o.alignment || (o.alignment & (o.alignment - 1)) || o.owner.empty()
            || !owners.insert(o.owner).second || !devices.contains(o.device) || !a.bytes
            || a.name.empty() || !phase_valid(a.first) || !phase_valid(a.last)
            || a.first > a.last || a.kind < AllocationKind::weight
            || a.kind > AllocationKind::status
            || (a.kind == AllocationKind::weight) != a.weight.has_value())
            throw std::invalid_argument("media_owned_allocation");
    }
}

struct CandidateBuilder {
    const StageManifest& manifest;
    const MediaShape& shape;
    MediaResidencyPlan plan;
    std::size_t channel_tile;

    void tensor(
        DevicePeak& device,
        std::string name,
        AllocationKind kind,
        Bytes bytes,
        MediaPhase first,
        MediaPhase last,
        Bytes internal_padding = 0
    ) {
        if (!bytes)
            return;
        std::size_t slot = device.allocations.size();
        // Same-kind slots alias only across strictly disjoint inclusive phases.
        for (std::size_t i = 0; i < device.allocations.size(); ++i) {
            if (device.allocations[i].kind != kind || device.allocations[i].weight)
                continue;
            bool disjoint = true;
            for (const auto& live : plan.tensors)
                if (live.device == device.device && live.allocation == i
                    && !(last < live.first || first > live.last))
                    disjoint = false;
            if (disjoint) {
                slot = i;
                break;
            }
        }
        if (slot == device.allocations.size())
            device.allocations.push_back(
                {name, kind, bytes, first, last, {}, 0, 0, internal_padding}
            );
        else {
            auto& a = device.allocations[slot];
            if (bytes > a.bytes) {
                a.bytes = bytes;
                a.internal_padding = internal_padding;
            }
            a.first = std::min(a.first, first);
            a.last = std::max(a.last, last);
        }
        plan.tensors.push_back(
            {std::move(name), device.device, bytes, first, last, slot}
        );
    }

    void weights(DevicePeak& device, std::size_t shares) {
        Bytes conversion = 0, conversion_padding = 0;
        for (const auto& w : manifest.weights) {
            const auto plane_rows = w.rows / w.planes;
            std::size_t first = 0, count = plane_rows;
            if (w.partition == WeightPartition::heads
                || w.partition == WeightPartition::kv_heads) {
                const auto group = w.partition == WeightPartition::kv_heads
                                       ? shape.heads / shape.kv_heads
                                       : 1;
                first = device.first_head / group * shape.head_dim;
                count = device.heads / group * shape.head_dim;
            } else if (w.partition == WeightPartition::channels) {
                const auto units = plane_rows / w.alignment;
                const auto begin = mul(units, device.first_head) / shares;
                const auto end = mul(units, device.first_head + device.heads) / shares;
                first = begin * w.alignment;
                count = (end - begin) * w.alignment;
            }
            if (!count)
                continue;
            const auto tile_alignment =
                w.partition == WeightPartition::heads
                        || w.partition == WeightPartition::kv_heads
                    ? shape.head_dim
                    : w.alignment;
            const auto columns = std::min(
                count,
                std::max(tile_alignment, channel_tile / tile_alignment * tile_alignment)
            );
            for (std::size_t plane = 0; plane < w.planes; ++plane) {
                const auto row = plane * plane_rows + first;
                const auto bytes = mul(mul(count, w.columns), width(w.storage));
                plan.weights.push_back(
                    {w.name, w.identity, device.device, row, count, columns, bytes}
                );
                device.allocations.push_back(
                    {"weight/" + w.name + "/" + std::to_string(plane),
                     AllocationKind::weight,
                     bytes,
                     MediaPhase::prepare,
                     MediaPhase::complete,
                     w.identity,
                     row,
                     count}
                );
            }
            Bytes widened = 0;
            if (w.precision == ProjectionPrecision::precise_f64)
                widened = width(w.storage) == 2 ? 12 : 8;
            else if (w.precision == ProjectionPrecision::f32 && width(w.storage) == 2)
                widened = 4;
            // Sequential tiles reuse widening buffers; H3's compact->F32->F64
            // casts retain both intermediate arrays, separate from stored weights.
            const auto weights = mul(mul(columns, w.columns), widened);
            const auto inputs =
                mul(mul(plan.rows, w.columns),
                    w.precision == ProjectionPrecision::bf16_f32 ? 2 : widened);
            const auto outputs = w.precision == ProjectionPrecision::precise_f64
                                     ? mul(mul(plan.rows, columns), 8)
                                     : 0;
            const auto alignment = manifest.allocation_alignment;
            auto scratch = workspace_bytes({weights, inputs, outputs}, alignment);
            auto raw = add(add(weights, inputs), outputs);
            if (w.precision == ProjectionPrecision::precise_f64
                && width(w.storage) == 2) {
                const auto weight_elements = mul(columns, w.columns);
                const auto input_elements = mul(plan.rows, w.columns);
                scratch = workspace_bytes(
                    {mul(weight_elements, 4),
                     mul(weight_elements, 8),
                     mul(input_elements, 4),
                     mul(input_elements, 8),
                     outputs},
                    alignment
                );
            }
            if (w.precision == ProjectionPrecision::convrot) {
                const auto input_elements = mul(plan.rows, w.columns);
                const auto weight_codes = mul(columns, w.columns);
                const auto sums = mul(mul(plan.rows, columns), 4);
                raw =
                    add(add(mul(input_elements, 9), mul(plan.rows, 4)),
                        add(weight_codes, sums));
                scratch = workspace_bytes(
                    {mul(input_elements, 4),
                     mul(input_elements, 4),
                     input_elements,
                     mul(plan.rows, 4),
                     weight_codes,
                     sums},
                    alignment
                );
            }
            if (scratch > conversion) {
                conversion = scratch;
                conversion_padding = scratch - raw;
            }
        }
        tensor(
            device,
            "resident/projection-conversion",
            AllocationKind::conversion,
            conversion,
            MediaPhase::kv,
            MediaPhase::ff,
            conversion_padding
        );
    }

    void activations(DevicePeak& d, bool multiple) {
        for (const auto& t : manifest.tensors)
            tensor(d, t.name, t.kind, t.bytes, t.first, t.last);
        tensor(
            d,
            "resident/blas",
            AllocationKind::blas,
            manifest.blas_bytes,
            MediaPhase::prepare,
            MediaPhase::complete
        );
        tensor(
            d,
            "resident/status",
            AllocationKind::status,
            manifest.status_bytes,
            MediaPhase::prepare,
            MediaPhase::complete
        );
        if (!transformer(manifest.stage))
            return;
        const auto local_inner = mul(d.heads, shape.head_dim);
        const auto kv =
            mul(mul(shape.tokens,
                    mul(d.heads / (shape.heads / shape.kv_heads), shape.head_dim)),
                4);
        const auto norm = mul(mul(plan.rows, shape.hidden), 4);
        tensor(
            d,
            "resident/hidden",
            AllocationKind::activation,
            mul(mul(shape.tokens, shape.hidden), 4),
            MediaPhase::prepare,
            MediaPhase::complete
        );
        tensor(
            d,
            "resident/key",
            AllocationKind::activation,
            kv,
            MediaPhase::kv,
            MediaPhase::attention
        );
        tensor(
            d,
            "resident/value",
            AllocationKind::activation,
            kv,
            MediaPhase::kv,
            MediaPhase::attention
        );
        tensor(
            d,
            "resident/normalized",
            AllocationKind::activation,
            norm,
            MediaPhase::kv,
            MediaPhase::ff
        );
        const auto q = mul(mul(plan.queries, local_inner), 4);
        tensor(
            d,
            "resident/query",
            AllocationKind::activation,
            q,
            MediaPhase::attention,
            MediaPhase::attention
        );
        tensor(
            d,
            "resident/attended",
            AllocationKind::activation,
            q,
            MediaPhase::attention,
            MediaPhase::attention
        );
        tensor(
            d,
            "resident/attention-input",
            AllocationKind::activation,
            mul(mul(plan.queries, shape.attention_inner), 4),
            MediaPhase::attention,
            MediaPhase::attention
        );
        tensor(
            d,
            "resident/kv-output",
            AllocationKind::activation,
            mul(mul(plan.rows, local_inner), 8),
            MediaPhase::kv,
            MediaPhase::kv
        );
        tensor(
            d,
            "resident/residual-tile",
            AllocationKind::activation,
            norm,
            MediaPhase::attention,
            MediaPhase::ff
        );
        // Conservative full gate/up tiles bound paired local results and gathers.
        tensor(
            d,
            "resident/ff-gate-up",
            AllocationKind::activation,
            mul(mul(plan.rows, shape.ff), 8),
            MediaPhase::ff,
            MediaPhase::ff
        );
        tensor(
            d,
            "resident/ff-input",
            AllocationKind::activation,
            mul(mul(plan.rows, shape.ff), 4),
            MediaPhase::ff,
            MediaPhase::ff
        );
        // One head at a time: packed Q/K/V, output, scores and softmax state.
        const auto attention_width = manifest.precise_attention ? 8 : 4;
        const auto query = mul(mul(plan.queries, shape.head_dim), attention_width);
        const auto key = mul(mul(plan.keys, shape.head_dim), attention_width);
        const auto scores = mul(mul(plan.queries, plan.keys), attention_width);
        const auto state = mul(plan.queries, attention_width);
        const auto raw =
            add(add(mul(query, 2), mul(key, 2)), add(scores, mul(state, 2)));
        const auto workspace = workspace_bytes(
            {query, key, key, query, scores, state, state},
            manifest.allocation_alignment
        );
        tensor(
            d,
            "resident/attention-workspace",
            AllocationKind::conversion,
            workspace,
            MediaPhase::attention,
            MediaPhase::attention,
            workspace - raw
        );
        if (multiple)
            tensor(
                d,
                "resident/peer-tile",
                AllocationKind::peer,
                mul(mul(plan.rows,
                        std::max({shape.hidden, shape.attention_inner, shape.ff})),
                    4),
                MediaPhase::attention,
                MediaPhase::ff
            );
    }

    void peaks(DevicePeak& d) {
        for (const auto& a : d.allocations)
            d.arena_bytes =
                add(d.arena_bytes, aligned(a.bytes, manifest.allocation_alignment));
        for (int p = 0; p <= static_cast<int>(MediaPhase::complete); ++p) {
            PhasePeak peak;
            peak.phase = static_cast<MediaPhase>(p);
            std::set<std::size_t> slots;
            for (std::size_t i = 0; i < d.allocations.size(); ++i)
                if (d.allocations[i].weight)
                    slots.insert(i);
            for (const auto& live : plan.tensors)
                if (live.device == d.device && live.first <= peak.phase
                    && peak.phase <= live.last)
                    slots.insert(live.allocation);
            for (auto slot : slots) {
                const auto& a = d.allocations[slot];
                Bytes* component = nullptr;
                switch (a.kind) {
                case AllocationKind::weight:
                    component = &peak.weights;
                    break;
                case AllocationKind::activation:
                    component = &peak.activations;
                    break;
                case AllocationKind::conversion:
                    component = &peak.conversion;
                    break;
                case AllocationKind::blas:
                    component = &peak.blas;
                    break;
                case AllocationKind::peer:
                    component = &peak.peer;
                    break;
                case AllocationKind::status:
                    component = &peak.status_and_alignment;
                    break;
                }
                *component = add(*component, a.bytes - a.internal_padding);
                peak.status_and_alignment =
                    add(peak.status_and_alignment,
                        add(a.internal_padding,
                            aligned(a.bytes, manifest.allocation_alignment) - a.bytes));
            }
            peak.total =
                add(add(add(peak.weights, peak.activations),
                        add(peak.conversion, peak.blas)),
                    add(peak.peer, peak.status_and_alignment));
            d.phases.push_back(peak);
        }
    }
};

bool same_allocation(const MediaAllocation& a, const MediaAllocation& b) {
    return a.name == b.name && a.kind == b.kind && a.weight == b.weight
           && a.first_row == b.first_row && a.rows == b.rows && b.bytes >= a.bytes;
}

MediaResidencyPlan candidate(
    const StageManifest& m,
    const MediaShape& s,
    std::span<const DeviceBudget> budgets,
    std::span<const OwnedMediaAllocation> owned,
    std::size_t first_heads,
    std::size_t rows,
    std::size_t keys,
    std::size_t columns
) {
    CandidateBuilder builder{m, s, {}, columns};
    auto& plan = builder.plan;
    plan.stage = m.stage;
    if (transformer(m.stage))
        plan.hidden_write_phases = {MediaPhase::attention, MediaPhase::ff};
    plan.rows = rows;
    plan.queries = std::min<std::size_t>(rows, 128);
    plan.keys = keys;
    plan.assumptions = {
        "Include explicit consumed scales, rotations, adapters and vectors.",
        "Reserve every arena slot; dormant buffers remain physically allocated.",
        "Complete K/V before hidden overwrite; FF is token-local.",
        "Gather full tiles before output-channel projections; direct peers only.",
        "Caller supplies actual operator, boundary, BLAS, status and context costs.",
        "Fit requires retirement, reprobe, reservation, then allocation."
    };
    const auto shares = transformer(m.stage) ? s.heads : 1;
    std::size_t first = 0;
    for (std::size_t i = 0; i < budgets.size(); ++i) {
        DevicePeak d;
        d.device = budgets[i].device;
        d.first_head = first;
        d.heads = budgets.size() == 1 ? shares
                                      : (i == 0 ? first_heads : shares - first_heads);
        builder.weights(d, shares);
        builder.activations(d, budgets.size() > 1);
        builder.peaks(d);
        first += d.heads;
        plan.devices.push_back(std::move(d));
    }
    for (auto& d : plan.devices) {
        const auto b =
            *std::find_if(budgets.begin(), budgets.end(), [&](const auto& budget) {
                return budget.device == d.device;
            });
        Bytes physical = b.physical_free, ledger = b.ledger_remaining;
        std::set<std::string> matched;
        for (std::size_t index = 0; index < d.allocations.size(); ++index) {
            const auto& a = d.allocations[index];
            const auto bytes = aligned(a.bytes, m.allocation_alignment);
            const auto reusable =
                std::find_if(owned.begin(), owned.end(), [&](const auto& o) {
                    return o.device == d.device && !matched.contains(o.owner)
                           && same_allocation(a, o.allocation)
                           && o.allocation.bytes >= bytes
                           && o.alignment >= m.allocation_alignment;
                });
            if (reusable == owned.end()) {
                d.new_physical = add(d.new_physical, bytes);
                d.new_ledger = add(d.new_ledger, bytes);
            } else {
                matched.insert(reusable->owner);
                plan.reuse.push_back({reusable->owner, d.device, index});
                d.new_ledger =
                    add(d.new_ledger,
                        bytes > reusable->ledger_bytes ? bytes - reusable->ledger_bytes
                                                       : 0);
            }
        }
        // Obsolete identities or geometries cannot help this candidate. Retire
        // every unmatched weight before evaluating either admission axis.
        for (const auto& o : owned) {
            if (o.device == d.device && o.allocation.kind == AllocationKind::weight
                && !matched.contains(o.owner)) {
                physical = add(physical, o.allocation.bytes);
                ledger = add(ledger, o.ledger_bytes);
                plan.retire_owners.push_back(o.owner);
            }
        }
        const auto overhead = add(b.context_unaccounted, b.safety_margin);
        d.new_physical = add(d.new_physical, overhead);
        d.new_ledger = add(d.new_ledger, overhead);
        d.physical_shortfall =
            d.new_physical > physical ? d.new_physical - physical : 0;
        d.ledger_shortfall = d.new_ledger > ledger ? d.new_ledger - ledger : 0;
    }
    return std::move(plan);
}

Bytes shortfall(const MediaResidencyPlan& plan) {
    Bytes shortage = 0;
    for (const auto& d : plan.devices)
        shortage = add(shortage, std::max(d.physical_shortfall, d.ledger_shortfall));
    return shortage;
}
}  // namespace

MediaPlanResult plan_media_stage(
    const StageManifest& m,
    const MediaShape& s,
    std::span<const DeviceBudget> budgets,
    const PeerMatrix& peers,
    std::span<const OwnedMediaAllocation> owned
) {
    try {
        validate(m, s, budgets, peers, owned);
        std::vector<DeviceBudget> ordered(budgets.begin(), budgets.end());
        std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
            return a.device < b.device;
        });
        budgets = ordered;
        if (!m.operators_supported || !m.workspaces_complete || budgets.size() > 2)
            return MediaPlanError{
                PlanFailure::unsupported_layout, "media_supported_layout", {}
            };
        for (const auto& w : m.weights)
            if ((w.storage == TensorStorage::i8_convrot
                 && (w.precision != ProjectionPrecision::convrot || w.alignment % 4
                     || (w.rotation_group != 64 && w.rotation_group != 256)
                     || w.columns % w.rotation_group
                     || w.columns > 2147483647 / (128 * 127)))
                || (w.precision == ProjectionPrecision::convrot
                    && w.storage != TensorStorage::i8_convrot)
                || (w.precision == ProjectionPrecision::bf16_f32
                    && w.storage != TensorStorage::bf16_exact)
                || (w.storage == TensorStorage::u8
                    && w.partition != WeightPartition::replicated))
                return MediaPlanError{
                    PlanFailure::unsupported_layout, "media_storage_compute_layout", {}
                };
        std::optional<MediaResidencyPlan> rejected;
        auto attempt = [&](std::span<const DeviceBudget> selected,
                           std::size_t heads) -> std::optional<MediaResidencyPlan> {
            const auto max_rows =
                transformer(m.stage) ? std::min<std::size_t>(256, s.tokens) : 1;
            const auto max_keys =
                transformer(m.stage) ? std::min<std::size_t>(4096, s.tokens) : 1;
            auto minimum = candidate(m, s, selected, owned, heads, 1, 1, 1);
            const auto shortage = shortfall(minimum);
            if (shortage) {
                if (!rejected || shortage < shortfall(*rejected))
                    rejected = std::move(minimum);
                return {};
            }
            // Demand is monotone in tile geometry. Skip a row/column pair when
            // even its one-key tile cannot fit; enumerate only legal fitting pairs.
            for (auto rows = max_rows; rows; rows /= 2)
                for (auto columns = std::size_t{256}; columns; columns /= 2) {
                    auto smallest_key =
                        candidate(m, s, selected, owned, heads, rows, 1, columns);
                    if (shortfall(smallest_key))
                        continue;
                    for (auto keys = max_keys; keys; keys /= 2) {
                        auto plan = candidate(
                            m, s, selected, owned, heads, rows, keys, columns
                        );
                        if (!shortfall(plan))
                            return plan;
                    }
                }
            return {};
        };
        for (std::size_t i = 0; i < budgets.size(); ++i)
            if (auto plan = attempt(budgets.subspan(i, 1), s.heads))
                return std::move(*plan);
        if (budgets.size() == 2 && transformer(m.stage) && s.kv_heads > 1) {
            const auto direct = [&](int from, int to) {
                return std::any_of(peers.begin(), peers.end(), [&](const auto& edge) {
                    return edge.from == from && edge.to == to && edge.direct;
                });
            };
            if (!direct(budgets[0].device, budgets[1].device)
                || !direct(budgets[1].device, budgets[0].device))
                return MediaPlanError{
                    PlanFailure::topology,
                    "media_direct_peer_required_both_directions",
                    {}
                };
            const auto group = s.heads / s.kv_heads;
            const std::array<Bytes, 2> capacities{
                std::min(budgets[0].ledger_remaining, budgets[0].physical_free),
                std::min(budgets[1].ledger_remaining, budgets[1].physical_free)
            };
            const auto target = std::clamp(
                weight_boundary(s.kv_heads, capacities, 1),
                std::size_t{1},
                s.kv_heads - 1
            );
            if (auto plan = attempt(budgets, target * group))
                return std::move(*plan);
            // Enumerate around the weighted share without O(heads) host metadata.
            // Equal distances visit the smaller split first, independent of input
            // order.
            const auto radius = std::max(target - 1, s.kv_heads - 1 - target);
            for (std::size_t distance = 1; distance <= radius; ++distance) {
                if (distance < target)
                    if (auto plan = attempt(budgets, (target - distance) * group))
                        return std::move(*plan);
                if (distance < s.kv_heads - target)
                    if (auto plan = attempt(budgets, (target + distance) * group))
                        return std::move(*plan);
            }
        }
        if (budgets.size() == 2 && transformer(m.stage) && s.kv_heads == 1)
            return MediaPlanError{
                PlanFailure::unsupported_layout, "media_unsplittable_kv_group", {}
            };
        if (budgets.size() == 2 && !transformer(m.stage))
            return MediaPlanError{
                PlanFailure::unsupported_layout,
                "media_decoder_multi_device_schedule",
                {}
            };
        return MediaPlanError{
            PlanFailure::capacity, "media_resident_arena_shortfall", std::move(rejected)
        };
    } catch (const std::overflow_error& error) {
        return MediaPlanError{PlanFailure::invalid_shape, error.what(), {}};
    } catch (const std::invalid_argument& error) {
        return MediaPlanError{PlanFailure::invalid_shape, error.what(), {}};
    }
}
}  // namespace kadan
