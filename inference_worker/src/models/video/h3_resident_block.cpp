#include "h3_resident_block.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

#include "kadan/cleanup_error.hpp"

namespace kadan::video::detail {
namespace {
void require(bool value, const char* why) {
    if (!value)
        throw std::runtime_error(why);
}

void stop(const std::atomic_bool& flag) {
    require(!flag.load(), "h3_denoiser_cancelled");
}

void add_projection(
    StageManifest& m,
    const H3Projection& p,
    std::size_t rows,
    std::size_t columns,
    WeightPartition partition,
    std::size_t planes,
    std::size_t tokens,
    std::size_t& rank,
    std::size_t& elements
) {
    m.weights.push_back(
        {p.weight,
         p.weight_id,
         TensorStorage::i8_convrot,
         rows,
         columns,
         partition,
         4,
         planes,
         ProjectionPrecision::convrot,
         p.group}
    );
    m.weights.push_back(
        {p.scale,
         p.scale_id,
         TensorStorage::f32,
         rows,
         1,
         partition,
         4,
         planes,
         ProjectionPrecision::f32}
    );
    if (!p.rank)
        return;
    m.weights.push_back(
        {p.lora_a,
         p.lora_a_id,
         TensorStorage::bf16_exact,
         p.rank,
         columns,
         WeightPartition::replicated,
         1,
         1,
         ProjectionPrecision::precise_f64}
    );
    m.weights.push_back(
        {p.lora_b,
         p.lora_b_id,
         TensorStorage::bf16_exact,
         rows,
         p.rank,
         partition,
         4,
         planes,
         ProjectionPrecision::precise_f64}
    );
    rank = std::max(rank, p.rank);
    elements = std::max(
        elements,
        p.rank * columns + rows / planes * p.rank
            + std::min(tokens, h3_tile_rows) * columns
    );
}

// Modulation and token groups live through the stage; LoRA scratch is used by
// every projection phase.
void add_shared_tensors(
    StageManifest& m,
    std::size_t hidden,
    std::size_t tokens,
    std::size_t groups,
    std::size_t rank,
    std::size_t elements,
    MediaPhase first
) {
    m.blas_bytes = h3_resident_blas_bytes;
    m.status_bytes = sizeof(std::uint32_t);
    m.tensors.push_back(
        {"block/modulation", groups * h3_modulation_rows * hidden * sizeof(float)}
    );
    m.tensors.push_back({"block/token-groups", tokens * sizeof(std::uint32_t)});
    if (!rank)
        return;
    m.tensors.push_back(
        {"lora/low",
         std::min(tokens, h3_tile_rows) * rank * sizeof(double),
         first,
         MediaPhase::ff,
         AllocationKind::conversion}
    );
    m.tensors.push_back(
        {"lora/weights",
         elements * sizeof(double),
         first,
         MediaPhase::ff,
         AllocationKind::conversion}
    );
}

// A device's share of one projection plane and its session views. Every plane of
// a projection holds the same relative share.
struct Plane {
    std::size_t first = 0, rows = 0;
    DeviceTensorView weight, scale, lora_b;
};

struct Projection {
    const H3Projection* spec = nullptr;
    std::size_t columns = 0;
    std::vector<Plane> planes;
    DeviceTensorView lora_a;
};

Projection bind(
    MediaSession& session,
    int device,
    const H3Projection& spec,
    std::size_t planes,
    std::size_t plane_rows,
    std::size_t columns
) {
    Projection p;
    p.spec = &spec;
    p.columns = columns;
    p.planes.resize(planes);
    auto slice = [&](const std::string& name, std::size_t plane) -> const WeightSlice& {
        for (const auto& w : session.plan().weights)
            if (w.device == device && w.name == name
                && w.first_row >= plane * plane_rows
                && w.first_row < (plane + 1) * plane_rows)
                return w;
        throw std::runtime_error("h3_resident_weight");
    };
    for (std::size_t i = 0; i < planes; ++i) {
        const auto& w = slice(spec.weight, i);
        const auto& s = slice(spec.scale, i);
        auto& plane = p.planes[i];
        plane.first = w.first_row - i * plane_rows;
        plane.rows = w.rows;
        require(
            s.first_row == w.first_row && s.rows == w.rows
                && plane.first + plane.rows <= plane_rows
                && plane.first == p.planes[0].first && plane.rows == p.planes[0].rows,
            "h3_resident_partition"
        );
        plane.weight = session.weight(w);
        plane.scale = session.weight(s);
        if (!spec.rank)
            continue;
        const auto& b = slice(spec.lora_b, i);
        require(
            b.first_row == w.first_row && b.rows == w.rows, "h3_resident_partition"
        );
        plane.lora_b = session.weight(b);
    }
    if (spec.rank)
        p.lora_a = session.weight(slice(spec.lora_a, 0));
    return p;
}

void ready(Projection& p, std::vector<DeviceTensorView*>& views) {
    for (auto& plane : p.planes) {
        views.push_back(&plane.weight);
        views.push_back(&plane.scale);
        if (p.spec->rank)
            views.push_back(&plane.lora_b);
    }
    if (p.spec->rank)
        views.push_back(&p.lora_a);
}

// One device: its launcher, the block's weight views, stage boundaries and
// reusable scratch. Each device keeps its own full hidden copy.
struct Lane {
    H3ResidentLauncher* launcher = nullptr;
    int id = -1;
    Projection qkv, out, fc1, fc2;
    std::array<DeviceTensorView, 4> norm;
    DeviceTensorView hidden, modulation, groups, rotary, normalized, key, value;
    DeviceTensorView conversion, lora_weights, blas, status;
};

std::vector<Lane>
lanes(MediaSession& session, std::span<H3ResidentLauncher* const> launchers) {
    const auto& plan = session.plan();
    require(!plan.devices.empty(), "h3_resident_plan");
    std::vector<Lane> result(plan.devices.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        auto& lane = result[i];
        lane.id = plan.devices[i].device;
        for (auto* candidate : launchers) {
            require(candidate, "h3_resident_launcher");
            if (candidate->device() == lane.id) {
                require(!lane.launcher, "h3_resident_launcher");
                lane.launcher = candidate;
            }
        }
        require(lane.launcher, "h3_resident_launcher");
    }
    return result;
}

DeviceTensorView matrix(
    MediaSession& session,
    int device,
    const char* name,
    std::size_t rows,
    std::size_t columns,
    DeviceDType type = DeviceDType::f32
) {
    return session.tensor(device, name, type, {rows, columns}, {columns, 1});
}

void bind_boundaries(
    MediaSession& session, Lane& lane, std::size_t tokens, std::size_t hidden
) {
    const auto& tensors = session.plan().tensors;
    const auto modulation =
        std::find_if(tensors.begin(), tensors.end(), [&](const auto& t) {
            return t.device == lane.id && t.name == "block/modulation";
        });
    require(modulation != tensors.end(), "h3_resident_plan");
    lane.hidden = matrix(session, lane.id, "resident/hidden", tokens, hidden);
    lane.modulation = matrix(
        session,
        lane.id,
        "block/modulation",
        modulation->bytes / sizeof(float) / hidden,
        hidden
    );
    lane.groups =
        session.tensor(lane.id, "block/token-groups", DeviceDType::u32, {tokens}, {1});
}

// Scratch shared by every launch in the current projection phases.
void bind_scratch(MediaSession& session, Lane& lane) {
    const auto& tensors = session.plan().tensors;
    const auto conversion =
        std::find_if(tensors.begin(), tensors.end(), [&](const auto& t) {
            return t.device == lane.id && t.name == "resident/projection-conversion";
        });
    require(conversion != tensors.end(), "h3_resident_plan");
    lane.conversion = session.tensor(
        lane.id,
        "resident/projection-conversion",
        DeviceDType::u8,
        {std::size_t(conversion->bytes)},
        {1}
    );
    lane.blas = session.tensor(
        lane.id, "resident/blas", DeviceDType::u8, {h3_resident_blas_bytes}, {1}
    );
    lane.status =
        session.tensor(lane.id, "resident/status", DeviceDType::u32, {1}, {1});
    std::size_t elements = 0;
    for (const auto* p : {&lane.qkv, &lane.out, &lane.fc1, &lane.fc2})
        if (p->spec && p->spec->rank)
            elements = std::max(
                elements,
                p->spec->rank * p->columns + p->planes[0].rows * p->spec->rank
                    + session.plan().rows * p->columns
            );
    if (elements)
        lane.lora_weights =
            session.tensor(lane.id, "lora/weights", DeviceDType::f64, {elements}, {1});
}

// Launch on the lane's stream and certify every output with its status word,
// which must be the last output.
template <class Read, class Write, class Operation>
void run(
    MediaSession& session, Lane& lane, Read& read, Write& write, Operation operation
) {
    session.launch(lane.id, read, write, [&](void* stream, auto in, auto out) {
        operation(stream, in, out);
    });
    session.complete(write, lane.status);
}

void project(
    MediaSession& session,
    Lane& lane,
    Projection& p,
    std::size_t plane,
    DeviceTensorView& input,
    DeviceTensorView& output,
    std::size_t rows
) {
    auto& part = p.planes[plane];
    const auto& spec = *p.spec;
    std::optional<DeviceTensorView> low;
    std::vector<DeviceTensorView*> read{&input, &part.weight, &part.scale};
    std::vector<DeviceTensorView*> write{&output, &lane.conversion};
    if (spec.rank) {
        low = matrix(session, lane.id, "lora/low", rows, spec.rank, DeviceDType::f64);
        read.push_back(&p.lora_a);
        read.push_back(&part.lora_b);
        write.push_back(&*low);
        write.push_back(&lane.lora_weights);
    }
    write.push_back(&lane.blas);
    write.push_back(&lane.status);
    run(session, lane, read, write, [&](void* stream, auto in, auto out) {
        lane.launcher->convrot(stream, in, out, spec.group, spec.lora_scale);
    });
}

// RMS norm and AdaLN modulation of hidden rows into a normalized tile.
void normalize(
    MediaSession& session,
    Lane& lane,
    std::size_t norm,
    std::size_t first,
    std::size_t rows,
    std::size_t shift,
    DeviceTensorView& normalized
) {
    auto hidden = session.rows(lane.hidden, first, rows);
    const std::array read{&hidden, &lane.norm[norm], &lane.modulation, &lane.groups};
    const std::array write{&normalized, &lane.status};
    run(session, lane, read, write, [&](void* stream, auto in, auto out) {
        lane.launcher->normalize(stream, in, out, first, shift);
    });
}

void residual(
    MediaSession& session,
    Lane& lane,
    DeviceTensorView& projected,
    std::size_t first,
    std::size_t rows,
    std::size_t index
) {
    auto hidden = session.rows(lane.hidden, first, rows);
    const std::array read{&hidden, &projected, &lane.modulation, &lane.groups};
    const std::array write{&hidden, &lane.status};
    run(session, lane, read, write, [&](void* stream, auto in, auto out) {
        lane.launcher->gated_residual(stream, in, out, first, index);
    });
}

void rotary(
    MediaSession& session,
    Lane& lane,
    DeviceTensorView& x,
    std::size_t norm,
    std::size_t first
) {
    const std::array read{&x, &lane.norm[norm], &lane.rotary};
    const std::array write{&x, &lane.status};
    run(session, lane, read, write, [&](void* stream, auto in, auto out) {
        lane.launcher->qk_rotary(stream, in, out, first);
    });
}

// Devices ordered by their output share of one projection; shares must tile
// [0, total) exactly so ordered gathers rebuild the full width.
std::vector<Lane*>
ordered(std::vector<Lane>& all, Projection Lane::* member, std::size_t total) {
    std::vector<Lane*> result;
    result.reserve(all.size());
    for (auto& lane : all)
        result.push_back(&lane);
    std::sort(result.begin(), result.end(), [&](auto* a, auto* b) {
        return (a->*member).planes[0].first < (b->*member).planes[0].first;
    });
    std::size_t column = 0;
    for (auto* lane : result) {
        const auto& plane = (lane->*member).planes[0];
        require(plane.first == column && plane.rows, "h3_resident_partition");
        column += plane.rows;
    }
    require(column == total, "h3_resident_partition");
    return result;
}

// Gather each device's share into a full-width tile on every device. A single
// device consumes its own share directly.
std::vector<DeviceTensorView*> gather(
    MediaSession& session,
    std::vector<Lane>& all,
    std::span<DeviceTensorView> shares,
    std::span<DeviceTensorView> full,
    const std::vector<Lane*>& order,
    Projection Lane::* member,
    const char* name,
    std::size_t rows,
    std::size_t width
) {
    std::vector<DeviceTensorView*> result;
    for (std::size_t i = 0; i < all.size(); ++i) {
        if (all.size() == 1) {
            result.push_back(&shares[i]);
            continue;
        }
        full[i] = matrix(session, all[i].id, name, rows, width);
        for (auto* source : order)
            session.gather(
                shares[std::size_t(source - all.data())],
                full[i],
                (source->*member).planes[0].first
            );
        result.push_back(&full[i]);
    }
    return result;
}

// fc1 gate/value -> SiLU gate -> fc2 -> residual with modulation index 5, for one
// row tile whose normalized input is inputs[i] on device i.
void feed_forward_tile(
    MediaSession& session,
    std::vector<Lane>& all,
    const std::vector<Lane*>& gate_order,
    const std::vector<Lane*>& down_order,
    std::span<DeviceTensorView> inputs,
    std::size_t first,
    std::size_t rows,
    std::size_t hidden,
    std::size_t width
) {
    const auto count = all.size();
    std::vector<DeviceTensorView> pair(count), gate(count), value(count), joined(count),
        down(count), gathered(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto& lane = all[i];
        const auto local = lane.fc1.planes[0].rows;
        pair[i] = matrix(session, lane.id, "resident/ff-gate-up", 2 * rows, local);
        gate[i] = session.rows(pair[i], 0, rows);
        value[i] = session.rows(pair[i], rows, rows);
        project(session, lane, lane.fc1, 0, inputs[i], gate[i], rows);
        project(session, lane, lane.fc1, 1, inputs[i], value[i], rows);
        const std::array read{&gate[i], &value[i]};
        const std::array write{&gate[i], &lane.status};
        run(session, lane, read, write, [&](void* stream, auto in, auto out) {
            lane.launcher->silu_gate(stream, in, out);
        });
    }
    const auto activated = gather(
        session,
        all,
        gate,
        joined,
        gate_order,
        &Lane::fc1,
        "resident/ff-input",
        rows,
        width
    );
    for (std::size_t i = 0; i < count; ++i) {
        auto& lane = all[i];
        down[i] = matrix(
            session, lane.id, "resident/residual-tile", rows, lane.fc2.planes[0].rows
        );
        project(session, lane, lane.fc2, 0, *activated[i], down[i], rows);
    }
    const auto projected = gather(
        session,
        all,
        down,
        gathered,
        down_order,
        &Lane::fc2,
        "resident/peer-tile",
        rows,
        hidden
    );
    for (std::size_t i = 0; i < count; ++i)
        residual(session, all[i], *projected[i], first, rows, 5);
}

void download(MediaSession& session, std::vector<Lane>& all, std::span<float> hidden) {
    // Every device holds the same updated hidden; return the first copy.
    auto result = std::move(all.front().hidden);
    all.clear();
    session.advance(MediaPhase::complete);
    session.download_output(result, std::as_writable_bytes(hidden));
}

}  // namespace

void close_h3_resident(
    MediaSession& session, std::span<H3ResidentLauncher* const> launchers, bool abort
) {
    std::exception_ptr first;
    for (auto* launcher : launchers)
        try {
            launcher->release_stream();
        } catch (...) {
            if (!first)
                first = std::current_exception();
        }
    try {
        if (abort || first)
            session.abort();
        else
            session.finish();
    } catch (...) {
        if (first)
            throw OperationCleanupError(first, std::current_exception());
        throw;
    }
    if (first)
        std::rethrow_exception(first);
}

StageManifest h3_feed_forward_manifest(
    const H3FeedForward& ff, std::size_t tokens, std::size_t groups
) {
    StageManifest m;
    m.stage = MediaStage::denoise;
    m.schedule = MediaSchedule::feed_forward;
    std::size_t rank = 0, elements = 0;
    add_projection(
        m,
        ff.fc[0],
        2 * ff.width,
        ff.hidden,
        WeightPartition::channels,
        2,
        tokens,
        rank,
        elements
    );
    add_projection(
        m,
        ff.fc[1],
        ff.hidden,
        ff.width,
        WeightPartition::channels,
        1,
        tokens,
        rank,
        elements
    );
    add_shared_tensors(m, ff.hidden, tokens, groups, rank, elements, MediaPhase::ff);
    m.workspaces_complete = true;
    return m;
}

MediaShape h3_feed_forward_shape(const H3FeedForward& ff, std::size_t tokens) {
    return {tokens, ff.hidden, 0, 0, 0, ff.width};
}

StageManifest
h3_block_manifest(const H3Block& b, std::size_t tokens, std::size_t groups) {
    StageManifest m;
    m.stage = MediaStage::denoise;
    m.schedule = MediaSchedule::full_block;
    const auto hidden = b.ff.hidden, inner = b.heads * b.head_dim;
    for (std::size_t i = 0; i < b.norm.size(); ++i)
        m.weights.push_back(
            {b.norm[i],
             b.norm_id[i],
             TensorStorage::bf16_exact,
             i < 2 ? hidden : b.head_dim,
             1,
             WeightPartition::replicated,
             1,
             1,
             ProjectionPrecision::f32}
        );
    std::size_t rank = 0, elements = 0;
    add_projection(
        m, b.qkv, 3 * inner, hidden, WeightPartition::heads, 3, tokens, rank, elements
    );
    add_projection(
        m, b.out, hidden, inner, WeightPartition::channels, 1, tokens, rank, elements
    );
    add_projection(
        m,
        b.ff.fc[0],
        2 * b.ff.width,
        hidden,
        WeightPartition::channels,
        2,
        tokens,
        rank,
        elements
    );
    add_projection(
        m,
        b.ff.fc[1],
        hidden,
        b.ff.width,
        WeightPartition::channels,
        1,
        tokens,
        rank,
        elements
    );
    add_shared_tensors(m, hidden, tokens, groups, rank, elements, MediaPhase::kv);
    m.tensors.push_back({"block/rotary", tokens * 96 * sizeof(double)});
    const auto queries = std::min<std::size_t>(128, tokens),
               keys = std::min<std::size_t>(4096, tokens);
    m.tensors.push_back(
        {"attention/scores",
         h3_attention_scores_bytes(b.heads, queries, keys),
         MediaPhase::attention,
         MediaPhase::attention,
         AllocationKind::conversion}
    );
    m.tensors.push_back(
        {"attention/state",
         2 * b.heads * queries * sizeof(double),
         MediaPhase::attention,
         MediaPhase::attention,
         AllocationKind::conversion}
    );
    m.tensors.push_back(
        {"attention/wide",
         h3_attention_wide_bytes(b.heads, queries, keys),
         MediaPhase::attention,
         MediaPhase::attention,
         AllocationKind::conversion}
    );
    m.workspaces_complete = true;
    return m;
}

MediaShape h3_block_shape(const H3Block& b, std::size_t tokens) {
    const auto inner = b.heads * b.head_dim;
    return {tokens, b.ff.hidden, inner, b.heads, b.head_dim, b.ff.width, 0, b.heads};
}

void upload_h3_feed_forward_inputs(
    MediaSession& session,
    const H3FeedForward& ff,
    std::span<const float> normalized,
    std::span<const float> hidden,
    std::span<const float> modulation,
    std::span<const std::uint32_t> token_groups
) {
    const auto tokens = token_groups.size(), rows = modulation.size() / ff.hidden;
    require(
        tokens && rows && rows % h3_modulation_rows == 0
            && modulation.size() == rows * ff.hidden
            && normalized.size() == tokens * ff.hidden
            && hidden.size() == tokens * ff.hidden,
        "h3_resident_input_shape"
    );
    for (const auto& planned : session.plan().devices) {
        const auto d = planned.device;
        auto normalized_view =
            matrix(session, d, "resident/normalized", tokens, ff.hidden);
        session.upload_input(normalized_view, std::as_bytes(normalized));
        auto hidden_view = matrix(session, d, "resident/hidden", tokens, ff.hidden);
        session.upload_input(hidden_view, std::as_bytes(hidden));
        auto modulation_view = matrix(session, d, "block/modulation", rows, ff.hidden);
        session.upload_input(modulation_view, std::as_bytes(modulation));
        auto groups_view =
            session.tensor(d, "block/token-groups", DeviceDType::u32, {tokens}, {1});
        session.upload_input(groups_view, std::as_bytes(token_groups));
    }
}

void upload_h3_block_inputs(
    MediaSession& session,
    const H3Block& b,
    std::span<const float> hidden,
    std::span<const float> modulation,
    std::span<const std::uint32_t> token_groups,
    std::span<const double> rotary
) {
    const auto width = b.ff.hidden, tokens = token_groups.size(),
               rows = modulation.size() / width;
    require(
        tokens && rows && rows % h3_modulation_rows == 0
            && modulation.size() == rows * width && hidden.size() == tokens * width
            && rotary.size() == tokens * 96,
        "h3_resident_input_shape"
    );
    for (const auto& planned : session.plan().devices) {
        const auto d = planned.device;
        auto hidden_view = matrix(session, d, "resident/hidden", tokens, width);
        session.upload_input(hidden_view, std::as_bytes(hidden));
        auto modulation_view = matrix(session, d, "block/modulation", rows, width);
        session.upload_input(modulation_view, std::as_bytes(modulation));
        auto groups_view =
            session.tensor(d, "block/token-groups", DeviceDType::u32, {tokens}, {1});
        session.upload_input(groups_view, std::as_bytes(token_groups));
        auto rotary_view =
            matrix(session, d, "block/rotary", tokens, 96, DeviceDType::f64);
        session.upload_input(rotary_view, std::as_bytes(rotary));
    }
}

void execute_h3_feed_forward(
    MediaSession& session,
    std::span<H3ResidentLauncher* const> launchers,
    const H3FeedForward& ff,
    std::size_t tokens,
    std::span<float> hidden,
    const std::atomic_bool& cancel
) {
    const auto& plan = session.plan();
    require(
        plan.schedule == MediaSchedule::feed_forward && plan.rows && plan.rows <= tokens
            && hidden.size() == tokens * ff.hidden,
        "h3_resident_plan"
    );
    auto all = lanes(session, launchers);
    for (auto& lane : all) {
        lane.fc1 = bind(session, lane.id, ff.fc[0], 2, ff.width, ff.hidden);
        lane.fc2 = bind(session, lane.id, ff.fc[1], 1, ff.hidden, ff.width);
        bind_boundaries(session, lane, tokens, ff.hidden);
        lane.normalized =
            matrix(session, lane.id, "resident/normalized", tokens, ff.hidden);
        std::vector<DeviceTensorView*> views{
            &lane.hidden, &lane.modulation, &lane.groups, &lane.normalized
        };
        ready(lane.fc1, views);
        ready(lane.fc2, views);
        session.require_ready(views);
    }
    const auto gate_order = ordered(all, &Lane::fc1, ff.width),
               down_order = ordered(all, &Lane::fc2, ff.hidden);
    session.advance(MediaPhase::ff);
    for (auto& lane : all)
        bind_scratch(session, lane);
    for (std::size_t first = 0; first < tokens;) {
        stop(cancel);
        const auto rows = std::min(plan.rows, tokens - first);
        std::vector<DeviceTensorView> inputs;
        inputs.reserve(all.size());
        for (auto& lane : all)
            inputs.push_back(session.rows(lane.normalized, first, rows));
        feed_forward_tile(
            session,
            all,
            gate_order,
            down_order,
            inputs,
            first,
            rows,
            ff.hidden,
            ff.width
        );
        first += rows;
    }
    download(session, all, hidden);
}

void execute_h3_block(
    MediaSession& session,
    std::span<H3ResidentLauncher* const> launchers,
    const H3Block& b,
    std::size_t tokens,
    std::span<float> hidden,
    const std::atomic_bool& cancel
) {
    const auto& plan = session.plan();
    const auto width = b.ff.hidden, inner = b.heads * b.head_dim;
    require(
        plan.schedule == MediaSchedule::full_block && plan.rows && plan.rows <= tokens
            && plan.queries && plan.queries <= plan.rows && plan.keys
            && plan.keys <= tokens && hidden.size() == tokens * width,
        "h3_resident_plan"
    );
    auto all = lanes(session, launchers);
    for (std::size_t i = 0; i < all.size(); ++i) {
        auto& lane = all[i];
        lane.qkv = bind(session, lane.id, b.qkv, 3, inner, width);
        lane.out = bind(session, lane.id, b.out, 1, width, inner);
        lane.fc1 = bind(session, lane.id, b.ff.fc[0], 2, b.ff.width, width);
        lane.fc2 = bind(session, lane.id, b.ff.fc[1], 1, width, b.ff.width);
        require(
            lane.qkv.planes[0].first == plan.devices[i].first_head * b.head_dim
                && lane.qkv.planes[0].rows == plan.devices[i].heads * b.head_dim,
            "h3_resident_partition"
        );
        for (std::size_t n = 0; n < b.norm.size(); ++n) {
            const auto& slices = plan.weights;
            const auto w =
                std::find_if(slices.begin(), slices.end(), [&](const auto& s) {
                    return s.device == lane.id && s.name == b.norm[n];
                });
            require(w != slices.end(), "h3_resident_weight");
            lane.norm[n] = session.weight(*w);
        }
        bind_boundaries(session, lane, tokens, width);
        lane.rotary =
            matrix(session, lane.id, "block/rotary", tokens, 96, DeviceDType::f64);
        std::vector<DeviceTensorView*> views{
            &lane.hidden, &lane.modulation, &lane.groups, &lane.rotary
        };
        for (auto& norm : lane.norm)
            views.push_back(&norm);
        for (auto* p : {&lane.qkv, &lane.out, &lane.fc1, &lane.fc2})
            ready(*p, views);
        session.require_ready(views);
    }
    const auto head_order = ordered(all, &Lane::qkv, inner),
               out_order = ordered(all, &Lane::out, width),
               gate_order = ordered(all, &Lane::fc1, b.ff.width),
               down_order = ordered(all, &Lane::fc2, width);

    // K/V for every token before any hidden row changes.
    session.advance(MediaPhase::kv);
    for (auto& lane : all) {
        bind_scratch(session, lane);
        const auto local = lane.qkv.planes[0].rows;
        lane.key = matrix(session, lane.id, "resident/key", tokens, local);
        lane.value = matrix(session, lane.id, "resident/value", tokens, local);
    }
    for (std::size_t first = 0; first < tokens;) {
        stop(cancel);
        const auto rows = std::min(plan.rows, tokens - first);
        for (auto& lane : all) {
            auto normalized =
                matrix(session, lane.id, "resident/normalized", rows, width);
            normalize(session, lane, 0, first, rows, 0, normalized);
            auto key = session.rows(lane.key, first, rows);
            project(session, lane, lane.qkv, 1, normalized, key, rows);
            rotary(session, lane, key, 3, first);
            auto value = session.rows(lane.value, first, rows);
            project(session, lane, lane.qkv, 2, normalized, value, rows);
        }
        first += rows;
    }

    // Per query tile: Q, attention over resident K/V, out_proj, residual (gate 2).
    session.advance(MediaPhase::attention);
    for (std::size_t first = 0; first < tokens;) {
        stop(cancel);
        const auto rows = std::min(plan.queries, tokens - first);
        const auto count = all.size();
        std::vector<DeviceTensorView> attended(count), joined(count), projected(count),
            gathered(count);
        for (std::size_t i = 0; i < count; ++i) {
            auto& lane = all[i];
            const auto local = lane.qkv.planes[0].rows, heads = local / b.head_dim;
            auto normalized =
                matrix(session, lane.id, "resident/normalized", rows, width);
            normalize(session, lane, 0, first, rows, 0, normalized);
            auto query = matrix(session, lane.id, "resident/query", rows, local);
            project(session, lane, lane.qkv, 0, normalized, query, rows);
            rotary(session, lane, query, 2, first);
            attended[i] = matrix(session, lane.id, "resident/attended", rows, local);
            auto scores = session.tensor(
                lane.id,
                "attention/scores",
                DeviceDType::f64,
                {heads * rows * plan.keys},
                {1}
            );
            auto state = session.tensor(
                lane.id, "attention/state", DeviceDType::f64, {2 * heads * rows}, {1}
            );
            auto wide = session.tensor(
                lane.id,
                "attention/wide",
                DeviceDType::f64,
                {std::size_t(
                    h3_attention_wide_bytes(heads, rows, plan.keys) / sizeof(double)
                )},
                {1}
            );
            const std::array read{&query, &lane.key, &lane.value};
            const std::array write{
                &attended[i], &scores, &state, &wide, &lane.blas, &lane.status
            };
            run(session, lane, read, write, [&](void* stream, auto in, auto out) {
                lane.launcher->attention(stream, in, out, plan.keys);
            });
        }
        const auto inputs = gather(
            session,
            all,
            attended,
            joined,
            head_order,
            &Lane::qkv,
            "resident/attention-input",
            rows,
            inner
        );
        for (std::size_t i = 0; i < count; ++i) {
            auto& lane = all[i];
            projected[i] = matrix(
                session,
                lane.id,
                "resident/residual-tile",
                rows,
                lane.out.planes[0].rows
            );
            project(session, lane, lane.out, 0, *inputs[i], projected[i], rows);
        }
        const auto outputs = gather(
            session,
            all,
            projected,
            gathered,
            out_order,
            &Lane::out,
            "resident/peer-tile",
            rows,
            width
        );
        for (std::size_t i = 0; i < count; ++i)
            residual(session, all[i], *outputs[i], first, rows, 2);
        first += rows;
    }
    for (auto& lane : all) {
        lane.key = {};
        lane.value = {};
    }

    // Token-local FF from the attention-updated hidden.
    session.advance(MediaPhase::ff);
    for (std::size_t first = 0; first < tokens;) {
        stop(cancel);
        const auto rows = std::min(plan.rows, tokens - first);
        std::vector<DeviceTensorView> inputs;
        inputs.reserve(all.size());
        for (auto& lane : all) {
            inputs.push_back(
                matrix(session, lane.id, "resident/normalized", rows, width)
            );
            normalize(session, lane, 1, first, rows, 3, inputs.back());
        }
        feed_forward_tile(
            session, all, gate_order, down_order, inputs, first, rows, width, b.ff.width
        );
        first += rows;
    }
    download(session, all, hidden);
}
}  // namespace kadan::video::detail
