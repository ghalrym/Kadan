#include "kadan/media_residency.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>

using namespace kadan;

namespace {
void check(bool ok, std::source_location at = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error("check_line_" + std::to_string(at.line()));
}

constexpr Bytes GiB = 1024ULL * 1024 * 1024;

StageWeight weight(
    std::string name,
    std::size_t rows,
    std::size_t columns,
    TensorStorage storage = TensorStorage::bf16_exact,
    WeightPartition partition = WeightPartition::channels,
    std::size_t alignment = 1,
    std::size_t planes = 1,
    ProjectionPrecision precision = ProjectionPrecision::bf16_f32
) {
    static std::uint64_t identity = 1;
    const auto width =
        storage == TensorStorage::f32
            ? 4
            : (storage == TensorStorage::i8_convrot || storage == TensorStorage::u8
                   ? 1
                   : 2);
    WeightIdentity id;
    id.inode = identity++;
    id.payload_end = rows * columns * width;
    return {
        std::move(name),
        id,
        storage,
        rows,
        columns,
        partition,
        alignment,
        planes,
        precision,
        precision == ProjectionPrecision::convrot ? 64U : 0U
    };
}

StageManifest manifest() {
    StageManifest m;
    m.workspaces_complete = true;
    m.allocation_alignment = 1;
    m.weights.push_back(
        weight("qkv", 24, 8, TensorStorage::bf16_exact, WeightPartition::heads, 2, 3)
    );
    m.weights.push_back(
        weight("ff", 32, 8, TensorStorage::bf16_exact, WeightPartition::channels, 4, 2)
    );
    m.weights.push_back(weight(
        "norm",
        8,
        1,
        TensorStorage::f32,
        WeightPartition::replicated,
        1,
        1,
        ProjectionPrecision::f32
    ));
    return m;
}

MediaShape shape() {
    return {1, 8, 8, 4, 2, 16, 0, 4};
}

std::vector<DeviceBudget> budgets(Bytes bytes = GiB) {
    return {{0, bytes, bytes, 0, 0}};
}

PeerMatrix peers() {
    return {{0, 1, true}, {1, 0, true}};
}

MediaResidencyPlan resident(
    MediaPlanResult result, std::source_location at = std::source_location::current()
) {
    if (const auto* error = std::get_if<MediaPlanError>(&result))
        throw std::runtime_error(error->detail + "_line_" + std::to_string(at.line()));
    return std::get<MediaResidencyPlan>(std::move(result));
}

MediaPlanError failure(MediaPlanResult result, PlanFailure reason) {
    check(std::holds_alternative<MediaPlanError>(result));
    auto error = std::get<MediaPlanError>(std::move(result));
    check(error.reason == reason);
    check((reason == PlanFailure::capacity) == error.rejected.has_value());
    return error;
}

const TensorLiveRange&
tensor(const MediaResidencyPlan& p, std::string_view name, int device = 0) {
    const auto it =
        std::find_if(p.tensors.begin(), p.tensors.end(), [&](const auto& t) {
            return t.name == name && t.device == device;
        });
    check(it != p.tensors.end());
    return *it;
}

void check_aliases(const MediaResidencyPlan& p) {
    for (std::size_t i = 0; i < p.tensors.size(); ++i)
        for (std::size_t j = i + 1; j < p.tensors.size(); ++j) {
            const auto& a = p.tensors[i];
            const auto& b = p.tensors[j];
            if (a.device == b.device && a.allocation == b.allocation)
                check(a.last < b.first || b.last < a.first);
        }
    check(p.kv_complete_before_hidden_write == MediaPhase::kv);
    for (auto phase : p.hidden_write_phases)
        check(phase > p.kv_complete_before_hidden_write);
    check(tensor(p, "resident/key").last == MediaPhase::attention);
    check(tensor(p, "resident/hidden").last == MediaPhase::complete);
}

void exact_fit_and_validation() {
    auto m = manifest();
    auto s = shape();
    auto b = budgets();
    const auto minimum =
        failure(plan_media_stage(m, s, budgets(0), {}), PlanFailure::capacity)
            .rejected->devices[0]
            .new_physical;
    b[0].ledger_remaining = b[0].physical_free = minimum;
    const auto p = resident(plan_media_stage(m, s, b, {}));
    check(p.devices[0].new_physical == minimum);
    --b[0].physical_free;
    auto error = failure(plan_media_stage(m, s, b, {}), PlanFailure::capacity);
    check(
        error.rejected->devices[0].physical_shortfall == 1
        && error.rejected->devices[0].ledger_shortfall == 0
    );
    ++b[0].physical_free;
    --b[0].ledger_remaining;
    check(
        failure(plan_media_stage(m, s, b, {}), PlanFailure::capacity)
            .rejected->devices[0]
            .ledger_shortfall
        == 1
    );
    check_aliases(p);
    const auto& qkv = m.weights[0];
    Bytes stored = 0;
    for (const auto& slice : p.weights)
        if (slice.identity == qkv.identity)
            stored += slice.stored_bytes;
    check(stored == 24 * 8 * 2);
    s.tokens = 0;
    failure(plan_media_stage(m, s, budgets(), {}), PlanFailure::invalid_shape);
    s = shape();
    s.head_dim = 3;
    failure(plan_media_stage(m, s, budgets(), {}), PlanFailure::invalid_shape);
    s = shape();
    s.text_tokens = 2;
    failure(plan_media_stage(m, s, budgets(), {}), PlanFailure::invalid_shape);
    s = shape();
    s.heads = std::numeric_limits<std::size_t>::max();
    failure(plan_media_stage(m, s, budgets(), {}), PlanFailure::invalid_shape);
    m.tensors.push_back(
        {"overflow",
         std::numeric_limits<Bytes>::max(),
         MediaPhase::kv,
         MediaPhase::ff,
         AllocationKind::activation}
    );
    failure(plan_media_stage(m, shape(), budgets(), {}), PlanFailure::invalid_shape);
    m.tensors.clear();
    m.allocation_alignment = 3;
    failure(plan_media_stage(m, shape(), budgets(), {}), PlanFailure::invalid_shape);
    m.allocation_alignment = 1;
    m.weights[0].identity.payload_end--;
    failure(plan_media_stage(m, shape(), budgets(), {}), PlanFailure::invalid_shape);
    m = manifest();
    m.weights.push_back(m.weights[0]);
    failure(plan_media_stage(m, shape(), budgets(), {}), PlanFailure::invalid_shape);
    m = manifest();
    m.workspaces_complete = false;
    failure(
        plan_media_stage(m, shape(), budgets(0), {}), PlanFailure::unsupported_layout
    );
    m = manifest();
    m.operators_supported = false;
    failure(
        plan_media_stage(m, shape(), budgets(0), {}), PlanFailure::unsupported_layout
    );
    b = {{0, GiB, GiB, 0, 0}, {0, GiB, GiB, 0, 0}};
    failure(plan_media_stage(manifest(), shape(), b, {}), PlanFailure::invalid_shape);
}

void precision_and_lifetimes() {
    auto m = manifest();
    const auto initial = resident(plan_media_stage(m, shape(), budgets(), {}));
    m.weights[0].precision = ProjectionPrecision::precise_f64;
    const auto precise = resident(plan_media_stage(m, shape(), budgets(), {}));
    check(precise.devices[0].phases[0].weights == initial.devices[0].phases[0].weights);
    check(
        tensor(precise, "resident/projection-conversion").bytes
        == 8 * 8 * 12 + 8 * 12 + 8 * 8
    );
    auto conv = weight(
        "convrot",
        16,
        64,
        TensorStorage::i8_convrot,
        WeightPartition::channels,
        4,
        1,
        ProjectionPrecision::convrot
    );
    m.weights = {conv};
    auto p = resident(plan_media_stage(m, shape(), budgets(), {}));
    check(
        tensor(p, "resident/projection-conversion").bytes
        == 64 * 9 + 4 + 16 * 64 + 16 * 4
    );
    m.allocation_alignment = 256;
    auto aligned_plan = resident(plan_media_stage(m, shape(), budgets(), {}));
    const auto& aligned_conversion =
        tensor(aligned_plan, "resident/projection-conversion");
    check(aligned_conversion.bytes == 2304);  // Six separately aligned ConvRot slices.
    const auto& allocation =
        aligned_plan.devices[0].allocations[aligned_conversion.allocation];
    check(allocation.internal_padding == 636);
    check(aligned_plan.devices[0].phases[1].status_and_alignment >= 636);
    m.allocation_alignment = 1;
    m.weights[0].rotation_group = 16;
    failure(
        plan_media_stage(m, shape(), budgets(0), {}), PlanFailure::unsupported_layout
    );
    m.weights[0].rotation_group = 64;
    m.weights[0].columns = 63;
    m.weights[0].identity.payload_end = 16 * 63;
    failure(
        plan_media_stage(m, shape(), budgets(0), {}), PlanFailure::unsupported_layout
    );
    m = manifest();
    m.tensors = {
        {"conditioning",
         123,
         MediaPhase::prepare,
         MediaPhase::complete,
         AllocationKind::activation},
        {"latents",
         77,
         MediaPhase::prepare,
         MediaPhase::complete,
         AllocationKind::activation},
        {"early", 100, MediaPhase::prepare, MediaPhase::kv, AllocationKind::peer},
        {"late", 200, MediaPhase::attention, MediaPhase::ff, AllocationKind::peer}
    };
    p = resident(plan_media_stage(m, shape(), budgets(), {}));
    check(tensor(p, "early").allocation == tensor(p, "late").allocation);
    check(tensor(p, "conditioning").allocation != tensor(p, "latents").allocation);
    check(p.devices[0].phases[0].peer == 200);  // Reserved slot grows to maximum.
    check_aliases(p);
    m.tensors[2].last = MediaPhase::attention;
    p = resident(plan_media_stage(m, shape(), budgets(), {}));
    check(tensor(p, "early").allocation != tensor(p, "late").allocation);
}

void reuse_and_retirement() {
    auto m = manifest();
    auto p = resident(plan_media_stage(m, shape(), budgets(), {}));
    std::vector<OwnedMediaAllocation> owned;
    for (std::size_t i = 0; i < p.devices[0].allocations.size(); ++i) {
        auto a = p.devices[0].allocations[i];
        owned.push_back({std::to_string(i), 0, MediaStage::denoise, a, a.bytes});
    }
    p = resident(plan_media_stage(m, shape(), budgets(0), {}, owned));
    check(p.devices[0].new_physical == 0 && p.devices[0].new_ledger == 0);
    check(p.reuse.size() == owned.size());
    owned[0].ledger_bytes = 0;
    auto error = failure(
        plan_media_stage(m, shape(), budgets(0), {}, owned), PlanFailure::capacity
    );
    check(error.rejected->devices[0].physical_shortfall == 0);
    check(error.rejected->devices[0].ledger_shortfall > 0);
    owned[0].ledger_bytes = owned[0].allocation.bytes;
    auto spare = owned[0];
    spare.owner = "unrelated";
    spare.allocation.name = "unrelated";
    spare.allocation.kind = AllocationKind::activation;
    spare.allocation.weight.reset();
    spare.allocation.bytes = GiB;
    // An unrelated owned buffer does not augment either free-memory axis.
    error = failure(
        plan_media_stage(m, shape(), budgets(0), {}, std::span(&spare, 1)),
        PlanFailure::capacity
    );
    check(error.rejected->devices[0].physical_shortfall > 0);
    owned.push_back(owned[0]);
    failure(
        plan_media_stage(m, shape(), budgets(), {}, owned), PlanFailure::invalid_shape
    );
    owned.pop_back();
    auto old = owned[0];
    old.owner = "inactive-text";
    old.allocation.weight->inode += 9999;
    old.allocation.bytes = GiB;
    old.ledger_bytes = GiB;
    old.stage = MediaStage::text;
    p = resident(plan_media_stage(m, shape(), budgets(0), {}, std::span(&old, 1)));
    check(p.retire_owners == std::vector<std::string>{"inactive-text"});
    old.stage = MediaStage::denoise;  // Previous model, same kind of stage, also dead.
    check(
        resident(plan_media_stage(m, shape(), budgets(0), {}, std::span(&old, 1)))
            .retire_owners.size()
        == 1
    );
    m.stage = MediaStage::image_decode;
    m.weights.clear();
    m.tensors = {
        {"latent",
         4096,
         MediaPhase::prepare,
         MediaPhase::complete,
         AllocationKind::activation},
        {"shortcut", 2048, MediaPhase::kv, MediaPhase::ff, AllocationKind::activation}
    };
    p = resident(plan_media_stage(m, {1, 1, 0, 0, 0, 0, 0}, budgets(6144), {}));
    check(
        p.devices[0].arena_bytes == 6144 && p.weights.empty()
        && p.hidden_write_phases.empty()
    );
}

void partitioning_and_topology() {
    auto m = manifest();
    auto s = shape();
    const auto one =
        failure(plan_media_stage(m, s, budgets(0), {}), PlanFailure::capacity)
            .rejected->devices[0]
            .new_physical;
    std::vector<DeviceBudget> b{
        {0, one * 3 / 4, one * 3 / 4, 0, 0}, {1, one * 3 / 4, one * 3 / 4, 0, 0}
    };
    auto p = resident(plan_media_stage(m, s, b, peers()));
    check(p.devices.size() == 2);
    for (const auto& w : m.weights) {
        std::vector<std::size_t> coverage(w.rows);
        for (const auto& slice : p.weights)
            if (slice.identity == w.identity) {
                check(
                    slice.first_row % w.alignment == 0 && slice.rows % w.alignment == 0
                    && slice.tile_columns % w.alignment == 0
                );
                for (auto row = slice.first_row; row < slice.first_row + slice.rows;
                     ++row)
                    ++coverage.at(row);
            }
        for (auto count : coverage)
            check(count == (w.partition == WeightPartition::replicated ? 2U : 1U));
    }
    check(
        p.devices[0].first_head == 0 && p.devices[1].first_head == p.devices[0].heads
    );
    check(p.devices[0].heads + p.devices[1].heads == s.heads);
    auto boundary = b;
    for (const auto& d : p.devices)
        boundary[d.device].ledger_remaining = boundary[d.device].physical_free =
            d.new_physical;
    check(resident(plan_media_stage(m, s, boundary, peers())).devices.size() == 2);
    --boundary[0].physical_free;
    const auto shortage =
        failure(plan_media_stage(m, s, boundary, peers()), PlanFailure::capacity);
    check(shortage.rejected->devices.size() == 2);
    check(shortage.rejected->devices[0].physical_shortfall == 1);
    auto asymmetric = b;
    asymmetric[0].ledger_remaining = asymmetric[0].physical_free =
        b[0].physical_free * 19 / 20;
    auto asymmetric_plan = resident(plan_media_stage(m, s, asymmetric, peers()));
    check(asymmetric_plan.devices.size() == 2);
    std::reverse(asymmetric.begin(), asymmetric.end());
    auto asymmetric_reversed = resident(plan_media_stage(m, s, asymmetric, peers()));
    for (std::size_t i = 0; i < asymmetric_plan.devices.size(); ++i) {
        check(
            asymmetric_plan.devices[i].device == asymmetric_reversed.devices[i].device
        );
        check(asymmetric_plan.devices[i].heads == asymmetric_reversed.devices[i].heads);
        check(
            asymmetric_plan.devices[i].new_physical
            == asymmetric_reversed.devices[i].new_physical
        );
    }
    auto full = resident(plan_media_stage(m, s, budgets(), {}));
    std::vector<OwnedMediaAllocation> full_weights;
    Bytes released = 0;
    for (const auto& a : full.devices[0].allocations)
        if (a.kind == AllocationKind::weight) {
            full_weights.push_back({a.name, 0, MediaStage::denoise, a, a.bytes});
            released += a.bytes;
        }
    auto with_retained = b;
    with_retained[0].physical_free = with_retained[0].ledger_remaining =
        b[0].physical_free > released ? b[0].physical_free - released : 0;
    auto repartition =
        resident(plan_media_stage(m, s, with_retained, peers(), full_weights));
    check(!repartition.retire_owners.empty());
    std::reverse(b.begin(), b.end());
    auto reversed = resident(plan_media_stage(m, s, b, peers()));
    check(
        reversed.devices[0].device == p.devices[0].device
        && reversed.devices[0].arena_bytes == p.devices[0].arena_bytes
    );
    failure(plan_media_stage(m, s, b, {{0, 1, true}}), PlanFailure::topology);
    failure(
        plan_media_stage(m, s, b, {{0, 1, true}, {1, 0, false}}), PlanFailure::topology
    );
    b[0].ledger_remaining = b[0].physical_free = GiB;
    check(
        resident(plan_media_stage(m, s, b, {})).devices[0].device == 1
    );  // Single GPU does not need peers.
    b = {{0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}};
    failure(plan_media_stage(m, s, b, peers()), PlanFailure::capacity);
}

void concrete_profiles() {
    // Metadata only: representative consumed matrices from current H3 block
    // shapes, repeated for all 50 blocks. Header inventories include unused
    // branches; these fixtures deliberately do not claim a full consumed manifest.
    StageManifest m;
    m.workspaces_complete = true;
    m.precise_attention = true;
    m.blas_bytes = 8 * 1024 * 1024;
    m.status_bytes = 256;
    for (std::size_t layer = 0; layer < 50; ++layer) {
        const auto prefix = std::to_string(layer) + "/";
        m.weights.push_back(weight(
            prefix + "adaln",
            96768,
            2688,
            TensorStorage::i8_convrot,
            WeightPartition::channels,
            4,
            1,
            ProjectionPrecision::convrot
        ));
        m.weights.push_back(weight(
            prefix + "qkv",
            21504,
            5376,
            TensorStorage::i8_convrot,
            WeightPartition::heads,
            4,
            3,
            ProjectionPrecision::convrot
        ));
        m.weights.push_back(weight(
            prefix + "out",
            5376,
            7168,
            TensorStorage::i8_convrot,
            WeightPartition::channels,
            4,
            1,
            ProjectionPrecision::convrot
        ));
        m.weights.push_back(weight(
            prefix + "gate-up",
            28672,
            5376,
            TensorStorage::i8_convrot,
            WeightPartition::channels,
            4,
            2,
            ProjectionPrecision::convrot
        ));
        m.weights.push_back(weight(
            prefix + "down",
            5376,
            14336,
            TensorStorage::i8_convrot,
            WeightPartition::channels,
            4,
            1,
            ProjectionPrecision::convrot
        ));
        m.weights.push_back(weight(
            prefix + "lora-A",
            384,
            5376,
            TensorStorage::bf16_exact,
            WeightPartition::replicated,
            1,
            1,
            ProjectionPrecision::precise_f64
        ));
        m.weights.push_back(weight(
            prefix + "lora-B",
            21504,
            384,
            TensorStorage::bf16_exact,
            WeightPartition::heads,
            4,
            3,
            ProjectionPrecision::precise_f64
        ));
    }
    Bytes int8_inventory = 0;
    for (const auto& w : m.weights)
        if (w.storage == TensorStorage::i8_convrot)
            int8_inventory += w.identity.payload_end;
    check(
        int8_inventory == 32273203200ULL
    );  // Exact I8 inventory in capacity-evidence.json.
    m.tensors = {
        {"conditioning",
         128 * 5120 * 4,
         MediaPhase::prepare,
         MediaPhase::complete,
         AllocationKind::activation},
        {"lora-rank-tile",
         256 * 384 * 4,
         MediaPhase::kv,
         MediaPhase::ff,
         AllocationKind::activation}
    };
    std::vector<DeviceBudget> b{
        {0, 24 * GiB, 24 * GiB, 384 * 1024 * 1024, 512 * 1024 * 1024},
        {1, 24 * GiB, 24 * GiB, 384 * 1024 * 1024, 512 * 1024 * 1024}
    };
    for (auto n : {5118U, 37838U, 109190U}) {
        auto p = resident(
            plan_media_stage(m, {n, 5376, 7168, 56, 128, 14336, 128, 56}, b, peers())
        );
        check_aliases(p);
        std::cout << "H3 " << n << " tokens:";
        for (const auto& d : p.devices)
            std::cout << " device " << d.device << " heads " << d.heads << " peak "
                      << d.new_physical;
        std::cout << " bytes (partial floating/adapter inventory)\n";
        for (const auto& d : p.devices) {
            const auto persistent =
                Bytes(n) * 5376 * 4 + 2 * Bytes(n) * d.heads * 128 * 4;
            check(d.phases[2].activations >= persistent);
            check(d.new_physical <= 24 * GiB);
        }
    }
    auto full_float = m;
    for (auto& w : full_float.weights)
        if (w.storage == TensorStorage::i8_convrot) {
            w.storage = TensorStorage::bf16_exact;
            w.precision = ProjectionPrecision::precise_f64;
            w.rotation_group = 0;
            w.identity.payload_end *= 2;
        }
    auto too_large = failure(
        plan_media_stage(
            full_float, {109190, 5376, 7168, 56, 128, 14336, 128, 56}, b, peers()
        ),
        PlanFailure::capacity
    );
    check(too_large.rejected->devices.size() == 2);
    check(
        too_large.rejected->devices[0].physical_shortfall > 0
        || too_large.rejected->devices[1].physical_shortfall > 0
    );
    // More text consumes more resident state; prompt length is not replaced by
    // the example's 128-token assumption.
    auto p = resident(
        plan_media_stage(m, {109190, 5376, 7168, 56, 128, 14336, 128, 56}, b, peers())
    );
    auto longer = resident(
        plan_media_stage(m, {110086, 5376, 7168, 56, 128, 14336, 1024, 56}, b, peers())
    );
    Bytes before = 0, after = 0;
    for (const auto& d : p.devices)
        before += d.arena_bytes;
    for (const auto& d : longer.devices)
        after += d.arena_bytes;
    check(after > before);
    // Qwen maximum illustrated shape: exact F32 hidden + K/V arithmetic.
    m = manifest();
    m.weights.clear();
    m.allocation_alignment = 256;
    m.weights.push_back(weight(
        "qwen-qkv",
        3 * 4096,
        4096,
        TensorStorage::bf16_exact,
        WeightPartition::heads,
        128,
        3
    ));
    p = resident(plan_media_stage(
        m, {18432, 4096, 4096, 32, 128, 12288, 2048, 32}, budgets(24 * GiB), {}
    ));
    check(
        tensor(p, "resident/hidden").bytes + tensor(p, "resident/key").bytes
            + tensor(p, "resident/value").bytes
        == 905969664
    );
    m.stage = MediaStage::text;
    m.weights.clear();
    m.weights.push_back(weight(
        "qwen-q", 4096, 4096, TensorStorage::bf16_exact, WeightPartition::heads, 128
    ));
    m.weights.push_back(weight(
        "qwen-kv",
        2 * 1024,
        4096,
        TensorStorage::bf16_exact,
        WeightPartition::kv_heads,
        128,
        2
    ));
    const MediaShape text{2048, 4096, 4096, 32, 128, 22016, 2048, 8};
    auto text_plan = resident(plan_media_stage(m, text, budgets(24 * GiB), {}));
    check(tensor(text_plan, "resident/key").bytes == 2048 * 1024 * 4);
    check(tensor(text_plan, "resident/value").bytes == 2048 * 1024 * 4);
    const auto low =
        failure(plan_media_stage(m, text, budgets(0), {}), PlanFailure::capacity)
            .rejected->devices[0]
            .new_physical;
    std::vector<DeviceBudget> text_budgets{
        {0, low * 3 / 4, low * 3 / 4, 0, 0}, {1, low * 3 / 4, low * 3 / 4, 0, 0}
    };
    text_plan = resident(plan_media_stage(m, text, text_budgets, peers()));
    check(text_plan.devices.size() == 2);
    for (const auto& d : text_plan.devices)
        check(d.first_head % 4 == 0 && d.heads % 4 == 0);
    std::cout << "Qwen representative peak " << p.devices[0].new_physical << " bytes\n";
}
}  // namespace

int main() {
    try {
        exact_fit_and_validation();
        precision_and_lifetimes();
        reuse_and_retirement();
        partitioning_and_topology();
        concrete_profiles();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
