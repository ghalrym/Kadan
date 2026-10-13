#pragma once
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "kadan/resources.hpp"
#include "kadan/weight_identity.hpp"

namespace kadan {
enum class MediaStage {
    text,
    denoise,
    image_decode,
    video_decode,
    audio_decode
};
enum class TensorStorage {
    bf16_exact,
    f16_exact,
    f32,
    i8_convrot,
    u8
};
enum class WeightPartition {
    replicated,
    channels,
    heads,
    kv_heads
};
enum class ProjectionPrecision {
    bf16_f32,
    f32,
    precise_f64,
    convrot
};
enum class PlanFailure {
    capacity,
    topology,
    unsupported_layout,
    invalid_shape
};
enum class MediaSchedule {
    full_block,
    feed_forward
};
// Inclusive lifetimes. Full blocks require complete K/V before hidden overwrite.
enum class MediaPhase {
    prepare,
    kv,
    attention,
    ff,
    complete
};
enum class AllocationKind {
    weight,
    activation,
    conversion,
    blas,
    peer,
    status
};

struct StageWeight {
    std::string name;
    WeightIdentity identity;
    TensorStorage storage = TensorStorage::f32;
    std::size_t rows = 0, columns = 0;
    WeightPartition partition = WeightPartition::channels;
    // Planar QKV/gate-up has independently partitioned planes, never an
    // interleaved head layout. Scales/adapters are separate weight identities.
    std::size_t alignment = 1, planes = 1;
    ProjectionPrecision precision = ProjectionPrecision::f32;
    std::size_t rotation_group = 0;
};

struct StageTensor {
    std::string name;
    Bytes bytes = 0;
    MediaPhase first = MediaPhase::prepare, last = MediaPhase::complete;
    // Replicated on participating devices: conditioning, latents, decoder
    // windows/overlap state and operator-specific workspaces.
    AllocationKind kind = AllocationKind::activation;
};

struct StageManifest {
    MediaStage stage = MediaStage::denoise;
    std::vector<StageWeight> weights;
    std::vector<StageTensor> tensors;
    Bytes blas_bytes = 0, status_bytes = 0;
    std::size_t allocation_alignment = 256;
    // A declaration by the manifest builder, not a probe: all operator-specific
    // scratch and boundary tensors are described. Missing evidence is unsupported.
    bool workspaces_complete = false;
    bool precise_attention = false, operators_supported = true;
    // FF consumes normalized input; nonzero attention_inner admits an output
    // projection in prepare with the same row buffer. No attention/KV execution.
    MediaSchedule schedule = MediaSchedule::full_block;
    // A consumed conditioning projection also needs conversion scratch in prepare.
    bool prepare_projection = false;
    // Actual input projections may initialize hidden before attention/FF.
    bool prepare_hidden = false;
};

struct MediaShape {
    std::size_t tokens = 0, hidden = 0, attention_inner = 0, heads = 0, head_dim = 0,
                ff = 0, text_tokens = 0, kv_heads = 0;
};

struct DeviceBudget {
    int device = -1;
    // Fresh free bytes AFTER context creation, excluding owned allocations.
    Bytes ledger_remaining = 0, physical_free = 0;
    Bytes context_unaccounted = 0, safety_margin = 0;
};

struct PeerEdge {
    int from = -1, to = -1;
    bool direct = false;
};

using PeerMatrix = std::vector<PeerEdge>;

struct MediaAllocation {
    std::string name;
    AllocationKind kind = AllocationKind::activation;
    Bytes bytes = 0;
    MediaPhase first = MediaPhase::prepare, last = MediaPhase::complete;
    std::optional<WeightIdentity> weight;
    std::size_t first_row = 0, rows = 0;
    Bytes internal_padding = 0;
    auto operator<=>(const MediaAllocation&) const = default;
};

struct OwnedMediaAllocation {
    // Globally unique ID: one physical allocation and one ledger charge.
    std::string owner;
    int device = -1;
    MediaStage stage = MediaStage::denoise;
    MediaAllocation allocation;
    Bytes ledger_bytes = 0;
    std::size_t alignment = 256;
};

struct WeightSlice {
    std::string name;
    WeightIdentity identity;
    int device = -1;
    std::size_t first_row = 0, rows = 0, tile_columns = 0;
    Bytes stored_bytes = 0;
};

struct TensorLiveRange {
    std::string name;
    int device = -1;
    Bytes bytes = 0;
    MediaPhase first = MediaPhase::prepare, last = MediaPhase::complete;
    std::size_t allocation = 0;
};

struct PhasePeak {
    MediaPhase phase = MediaPhase::prepare;
    Bytes weights = 0, activations = 0, conversion = 0, blas = 0, peer = 0,
          status_and_alignment = 0, total = 0;
};

struct DevicePeak {
    int device = -1;
    std::size_t first_head = 0, heads = 0;
    std::vector<MediaAllocation> allocations;
    std::vector<PhasePeak> phases;
    Bytes arena_bytes = 0, new_physical = 0, new_ledger = 0, physical_shortfall = 0,
          ledger_shortfall = 0;
};

struct AllocationReuse {
    std::string owner;
    int device = -1;
    std::size_t allocation = 0;
};

struct MediaResidencyPlan {
    MediaStage stage = MediaStage::denoise;
    MediaSchedule schedule = MediaSchedule::full_block;
    std::size_t rows = 0, queries = 0, keys = 0;
    std::vector<WeightSlice> weights;
    std::vector<TensorLiveRange> tensors;
    std::vector<DevicePeak> devices;
    std::vector<AllocationReuse> reuse;
    std::vector<std::string> retire_owners, assumptions;
    std::optional<MediaPhase> kv_complete_before_hidden_write = MediaPhase::kv;
    std::vector<MediaPhase> hidden_write_phases;
};

struct MediaPlanError {
    PlanFailure reason;
    std::string detail;
    // Least-short capacity candidate; topology/layout have no certificate.
    std::optional<MediaResidencyPlan> rejected;
};

using MediaPlanResult = std::variant<MediaResidencyPlan, MediaPlanError>;
// Pure metadata. One device, then aligned two-device shares and decreasing tiles.
// The next session MR must retire listed owners, reprobe, reserve, then allocate.
// Decoder workspaces are explicit manifest tensors; no decoder kernel is modeled.
MediaPlanResult plan_media_stage(
    const StageManifest&,
    const MediaShape&,
    std::span<const DeviceBudget>,
    const PeerMatrix&,
    std::span<const OwnedMediaAllocation> owned = {}
);
}  // namespace kadan
