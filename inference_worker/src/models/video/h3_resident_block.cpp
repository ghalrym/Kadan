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
    const bool convrot = p.storage == TensorStorage::i8_convrot;
    m.weights.push_back(
        {p.weight,
         p.weight_id,
         p.storage,
         rows,
         columns,
         partition,
         4,
         planes,
         convrot ? ProjectionPrecision::convrot : ProjectionPrecision::precise_f64,
         convrot ? p.group : 0}
    );
    if (convrot)
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
    if (!p.bias.empty())
        m.weights.push_back(
            {p.bias,
             p.bias_id,
             p.bias_storage,
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
    DeviceTensorView weight, scale, lora_b, bias;
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
        const auto& s = spec.scale.empty() ? w : slice(spec.scale, i);
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
        if (!spec.scale.empty())
            plane.scale = session.weight(s);
        if (!spec.bias.empty()) {
            const auto& b = slice(spec.bias, i);
            require(
                b.first_row == w.first_row && b.rows == w.rows, "h3_resident_partition"
            );
            plane.bias = session.weight(b);
        }
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
        if (!p.spec->scale.empty())
            views.push_back(&plane.scale);
        if (!p.spec->bias.empty())
            views.push_back(&plane.bias);
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
    Projection qkv, out, adaln, fc1, fc2, video, audio;
    std::array<DeviceTensorView, 4> norm;
    DeviceTensorView final_norm, time_ids;
    DeviceTensorView hidden, modulation, groups, rotary, time, normalized, key, value;
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
    const auto row = h3_modulation_tags * h3_modulation_rows * hidden;
    lane.modulation = matrix(
        session,
        lane.id,
        "block/modulation",
        modulation->bytes / sizeof(float) / row,
        row
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
    const bool convrot = spec.storage == TensorStorage::i8_convrot;
    std::vector<DeviceTensorView*> read{&input, &part.weight};
    if (convrot)
        read.push_back(&part.scale);
    std::vector<DeviceTensorView*> write{&output, &lane.conversion};
    if (!spec.bias.empty())
        read.push_back(&part.bias);
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
        if (convrot)
            lane.launcher->convrot(stream, in, out, spec.group, spec.lora_scale);
        else
            lane.launcher->dense(stream, in, out, spec.lora_scale);
    });
}

// RMS norm of hidden rows into `normalized`, modulated by the block's AdaLN table
// with `shift`. A plain norm (modulation null) may write the hidden rows in place.
void normalize(
    MediaSession& session,
    Lane& lane,
    DeviceTensorView& weight,
    std::size_t first,
    std::size_t rows,
    DeviceTensorView& normalized,
    DeviceTensorView* modulation = nullptr,
    DeviceTensorView* groups = nullptr,
    std::size_t shift = 0,
    std::size_t stride = h3_modulation_rows
) {
    auto hidden = session.rows(lane.hidden, first, rows);
    std::vector<DeviceTensorView*> read{&hidden, &weight};
    if (modulation) {
        read.push_back(modulation);
        read.push_back(groups);
    }
    // In place, the output is the same hidden rows the kernel reads.
    auto* output = &normalized == &lane.hidden ? &hidden : &normalized;
    const std::array write{output, &lane.status};
    run(session, lane, read, write, [&](void* stream, auto in, auto out) {
        lane.launcher->normalize(stream, in, out, first, shift, stride);
    });
}

void normalize(
    MediaSession& session,
    Lane& lane,
    std::size_t norm,
    std::size_t first,
    std::size_t rows,
    std::size_t shift,
    DeviceTensorView& normalized
) {
    normalize(
        session,
        lane,
        lane.norm[norm],
        first,
        rows,
        normalized,
        &lane.modulation,
        &lane.groups,
        shift
    );
}

void add(
    MediaSession& session,
    Lane& lane,
    DeviceTensorView& projected,
    std::size_t first,
    std::size_t rows
) {
    auto hidden = session.rows(lane.hidden, first, rows);
    const std::array read{&hidden, &projected};
    const std::array write{&hidden, &lane.status};
    run(session, lane, read, write, [&](void* stream, auto in, auto out) {
        lane.launcher->add(stream, in, out);
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

// Attention scratch for every head of one query tile against one key tile.
void add_attention_tensors(
    StageManifest& m,
    std::size_t heads,
    std::size_t queries,
    std::size_t keys,
    MediaPhase first,
    MediaPhase last
) {
    m.tensors.push_back(
        {"attention/scores",
         h3_attention_scores_bytes(heads, queries, keys),
         first,
         last,
         AllocationKind::conversion}
    );
    m.tensors.push_back(
        {"attention/state",
         2 * heads * queries * sizeof(double),
         first,
         last,
         AllocationKind::conversion}
    );
    m.tensors.push_back(
        {"attention/wide",
         h3_attention_wide_bytes(heads, queries, keys),
         first,
         last,
         AllocationKind::conversion}
    );
}

void add_norm(
    StageManifest& m,
    const std::string& name,
    const WeightIdentity& id,
    std::size_t rows
) {
    m.weights.push_back(
        {name,
         id,
         TensorStorage::bf16_exact,
         rows,
         1,
         WeightPartition::replicated,
         1,
         1,
         ProjectionPrecision::f32}
    );
}

DeviceTensorView norm_view(MediaSession& session, int device, const std::string& name) {
    const auto& slices = session.plan().weights;
    const auto w = std::find_if(slices.begin(), slices.end(), [&](const auto& s) {
        return s.device == device && s.name == name;
    });
    require(w != slices.end(), "h3_resident_weight");
    return session.weight(*w);
}

// One attention call for a query tile against every key, into `attended`.
void attend(
    MediaSession& session,
    Lane& lane,
    DeviceTensorView& query,
    DeviceTensorView& key,
    DeviceTensorView& value,
    DeviceTensorView& attended,
    std::size_t heads,
    std::size_t rows,
    std::size_t key_tile
) {
    auto scores = session.tensor(
        lane.id, "attention/scores", DeviceDType::f64, {heads * rows * key_tile}, {1}
    );
    auto state = session.tensor(
        lane.id, "attention/state", DeviceDType::f64, {2 * heads * rows}, {1}
    );
    auto wide = session.tensor(
        lane.id,
        "attention/wide",
        DeviceDType::f64,
        {std::size_t(h3_attention_wide_bytes(heads, rows, key_tile) / sizeof(double))},
        {1}
    );
    const std::array read{&query, &key, &value};
    const std::array write{&attended, &scores, &state, &wide, &lane.blas, &lane.status};
    run(session, lane, read, write, [&](void* stream, auto in, auto out) {
        lane.launcher->attention(stream, in, out, key_tile);
    });
}

void normalize_qk(
    MediaSession& session,
    Lane& lane,
    DeviceTensorView& x,
    std::size_t norm,
    std::size_t first
) {
    const std::array read{&x, &lane.norm[norm]};
    const std::array write{&x, &lane.status};
    run(session, lane, read, write, [&](void* stream, auto in, auto out) {
        lane.launcher->qk_rotary(stream, in, out, first);
    });
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

StageManifest h3_block_manifest(
    const H3Block& b, std::size_t tokens, std::size_t groups, bool retained
) {
    StageManifest m;
    m.stage = MediaStage::denoise;
    m.schedule = MediaSchedule::full_block;
    const auto hidden = b.ff.hidden, inner = b.heads * b.head_dim;
    for (std::size_t i = 0; i < b.norm.size(); ++i)
        add_norm(m, b.norm[i], b.norm_id[i], i < 2 ? hidden : b.head_dim);
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
    add_projection(
        m,
        b.adaln,
        h3_modulation_tags * h3_modulation_rows * hidden,
        b.time_width,
        WeightPartition::replicated,
        1,
        tokens,
        rank,
        elements
    );
    add_shared_tensors(m, hidden, tokens, groups, rank, elements, MediaPhase::kv);
    m.tensors.push_back({"block/rotary", tokens * 96 * sizeof(double)});
    m.tensors.push_back(
        {"block/time", groups / h3_modulation_tags * b.time_width * sizeof(float)}
    );
    if (retained)
        m.retained_inputs = {
            "resident/hidden", "block/token-groups", "block/rotary", "block/time"
        };
    const auto queries = std::min<std::size_t>(128, tokens),
               keys = std::min<std::size_t>(4096, tokens);
    add_attention_tensors(
        m, b.heads, queries, keys, MediaPhase::attention, MediaPhase::attention
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
    const auto row = h3_modulation_tags * h3_modulation_rows * ff.hidden;
    const auto tokens = token_groups.size(), rows = modulation.size() / row;
    require(
        tokens && rows && modulation.size() == rows * row
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
        auto modulation_view = matrix(session, d, "block/modulation", rows, row);
        session.upload_input(modulation_view, std::as_bytes(modulation));
        auto groups_view =
            session.tensor(d, "block/token-groups", DeviceDType::u32, {tokens}, {1});
        session.upload_input(groups_view, std::as_bytes(token_groups));
    }
}

void upload_h3_step_inputs(
    MediaSession& session,
    std::size_t width,
    std::size_t time_width,
    std::span<const float> hidden,
    std::span<const float> time,
    std::span<const std::uint32_t> token_groups,
    std::span<const double> rotary
) {
    const auto tokens = token_groups.size(), timesteps = time.size() / time_width;
    require(
        tokens && timesteps && timesteps <= 3 && time.size() == timesteps * time_width
            && hidden.size() == tokens * width && rotary.size() == tokens * 96,
        "h3_resident_input_shape"
    );
    for (const auto& planned : session.plan().devices) {
        const auto d = planned.device;
        auto hidden_view = matrix(session, d, "resident/hidden", tokens, width);
        session.upload_input(hidden_view, std::as_bytes(hidden));
        auto time_view = matrix(session, d, "block/time", timesteps, time_width);
        session.upload_input(time_view, std::as_bytes(time));
        auto groups_view =
            session.tensor(d, "block/token-groups", DeviceDType::u32, {tokens}, {1});
        session.upload_input(groups_view, std::as_bytes(token_groups));
        auto rotary_view =
            matrix(session, d, "block/rotary", tokens, 96, DeviceDType::f64);
        session.upload_input(rotary_view, std::as_bytes(rotary));
    }
}

void upload_h3_final_inputs(
    MediaSession& session, std::span<const std::uint32_t> time_ids
) {
    require(!time_ids.empty(), "h3_resident_input_shape");
    for (const auto& planned : session.plan().devices) {
        auto view = session.tensor(
            planned.device, "final/time-ids", DeviceDType::u32, {time_ids.size()}, {1}
        );
        session.upload_input(view, std::as_bytes(time_ids));
    }
}

void download_h3_hidden(
    MediaSession& session, std::size_t hidden_width, std::span<float> hidden
) {
    const auto& plan = session.plan();
    require(
        !plan.devices.empty() && hidden_width && hidden.size() % hidden_width == 0,
        "h3_resident_input_shape"
    );
    // Every device holds the same updated hidden; return the first copy.
    auto view = matrix(
        session,
        plan.devices.front().device,
        "resident/hidden",
        hidden.size() / hidden_width,
        hidden_width
    );
    session.download_output(view, std::as_writable_bytes(hidden));
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
    const std::atomic_bool& cancel
) {
    const auto& plan = session.plan();
    const auto width = b.ff.hidden, inner = b.heads * b.head_dim;
    require(
        plan.schedule == MediaSchedule::full_block && plan.rows && plan.rows <= tokens
            && plan.queries && plan.queries <= plan.rows && plan.keys
            && plan.keys <= tokens,
        "h3_resident_plan"
    );
    auto all = lanes(session, launchers);
    for (std::size_t i = 0; i < all.size(); ++i) {
        auto& lane = all[i];
        lane.qkv = bind(session, lane.id, b.qkv, 3, inner, width);
        lane.out = bind(session, lane.id, b.out, 1, width, inner);
        lane.fc1 = bind(session, lane.id, b.ff.fc[0], 2, b.ff.width, width);
        lane.fc2 = bind(session, lane.id, b.ff.fc[1], 1, width, b.ff.width);
        lane.adaln = bind(
            session,
            lane.id,
            b.adaln,
            1,
            h3_modulation_tags * h3_modulation_rows * width,
            b.time_width
        );
        require(
            lane.qkv.planes[0].first == plan.devices[i].first_head * b.head_dim
                && lane.qkv.planes[0].rows == plan.devices[i].heads * b.head_dim,
            "h3_resident_partition"
        );
        for (std::size_t n = 0; n < b.norm.size(); ++n)
            lane.norm[n] = norm_view(session, lane.id, b.norm[n]);
        bind_boundaries(session, lane, tokens, width);
        lane.rotary =
            matrix(session, lane.id, "block/rotary", tokens, 96, DeviceDType::f64);
        lane.time = matrix(
            session,
            lane.id,
            "block/time",
            lane.modulation.layout().dimensions[0],
            b.time_width
        );
        // Modulation is this block's AdaLN output, produced on the device below.
        std::vector<DeviceTensorView*> views{
            &lane.hidden, &lane.groups, &lane.rotary, &lane.time
        };
        for (auto& norm : lane.norm)
            views.push_back(&norm);
        for (auto* p : {&lane.qkv, &lane.out, &lane.adaln, &lane.fc1, &lane.fc2})
            ready(*p, views);
        session.require_ready(views);
    }
    const auto head_order = ordered(all, &Lane::qkv, inner),
               out_order = ordered(all, &Lane::out, width),
               gate_order = ordered(all, &Lane::fc1, b.ff.width),
               down_order = ordered(all, &Lane::fc2, width);

    // This block's AdaLN modulation, then K/V for every token before any hidden
    // row changes.
    session.advance(MediaPhase::kv);
    for (auto& lane : all) {
        bind_scratch(session, lane);
        project(
            session,
            lane,
            lane.adaln,
            0,
            lane.time,
            lane.modulation,
            lane.time.layout().dimensions[0]
        );
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
            attend(
                session,
                lane,
                query,
                lane.key,
                lane.value,
                attended[i],
                heads,
                rows,
                plan.keys
            );
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
    // Hidden stays resident for the next block's admission.
    all.clear();
    session.advance(MediaPhase::complete);
}

StageManifest h3_refiner_manifest(
    const H3Refiner& r,
    std::size_t tokens,
    std::size_t text,
    std::size_t timesteps,
    bool retained
) {
    require(
        text && text <= tokens && timesteps && timesteps <= 3, "h3_resident_input_shape"
    );
    StageManifest m;
    m.stage = MediaStage::denoise;
    m.schedule = MediaSchedule::feed_forward;
    const auto inner = r.heads * r.head_dim;
    for (std::size_t i = 0; i < r.norm.size(); ++i)
        add_norm(m, r.norm[i], r.norm_id[i], i < 2 ? r.hidden : r.head_dim);
    if (!r.final_norm.empty())
        add_norm(m, r.final_norm, r.final_norm_id, r.hidden);
    std::size_t rank = 0, elements = 0;
    const auto replicated = WeightPartition::replicated;
    add_projection(
        m, r.qkv, 3 * inner, r.hidden, replicated, 3, tokens, rank, elements
    );
    add_projection(m, r.out, r.hidden, inner, replicated, 1, tokens, rank, elements);
    add_projection(
        m, r.fc1, 2 * r.width, r.hidden, replicated, 2, tokens, rank, elements
    );
    add_projection(m, r.fc2, r.hidden, r.width, replicated, 1, tokens, rank, elements);
    // The stage carries the step's token groups, rotary and time embedding for the
    // blocks that follow; its modulation table is unused.
    add_shared_tensors(
        m,
        r.hidden,
        tokens,
        timesteps * h3_modulation_tags,
        rank,
        elements,
        MediaPhase::ff
    );
    m.tensors.push_back({"block/rotary", tokens * 96 * sizeof(double)});
    m.tensors.push_back({"block/time", timesteps * r.time_width * sizeof(float)});
    const auto queries = std::min<std::size_t>(128, text),
               keys = std::min<std::size_t>(4096, text);
    for (const auto* name : {"refiner/query", "refiner/key", "refiner/value"})
        m.tensors.push_back(
            {name, text * inner * sizeof(float), MediaPhase::ff, MediaPhase::ff}
        );
    m.tensors.push_back(
        {"refiner/attended",
         queries * inner * sizeof(float),
         MediaPhase::ff,
         MediaPhase::ff}
    );
    add_attention_tensors(m, r.heads, queries, keys, MediaPhase::ff, MediaPhase::ff);
    if (retained)
        m.retained_inputs = {
            "resident/hidden", "block/token-groups", "block/rotary", "block/time"
        };
    m.workspaces_complete = true;
    return m;
}

MediaShape h3_refiner_shape(const H3Refiner& r, std::size_t tokens) {
    return {tokens, r.hidden, 0, 0, 0, r.width};
}

StageManifest h3_final_manifest(
    const H3Final& f,
    std::size_t tokens,
    std::size_t video,
    std::size_t audio,
    std::size_t timesteps
) {
    require(
        timesteps && timesteps <= 3 && (video || audio) && video + audio <= tokens,
        "h3_resident_input_shape"
    );
    StageManifest m;
    m.stage = MediaStage::denoise;
    m.schedule = MediaSchedule::feed_forward;
    m.blas_bytes = h3_resident_blas_bytes;
    m.status_bytes = sizeof(std::uint32_t);
    add_norm(m, f.norm, f.norm_id, f.hidden);
    std::size_t rank = 0, elements = 0;
    const auto replicated = WeightPartition::replicated;
    add_projection(
        m,
        f.adaln,
        h3_final_modulation_rows * f.hidden,
        f.time_width,
        replicated,
        1,
        tokens,
        rank,
        elements
    );
    if (video)
        add_projection(
            m, f.video, f.video_width, f.hidden, replicated, 1, tokens, rank, elements
        );
    if (audio)
        add_projection(
            m, f.audio, f.audio_width, f.hidden, replicated, 1, tokens, rank, elements
        );
    m.tensors.push_back({"block/time", timesteps * f.time_width * sizeof(float)});
    m.tensors.push_back(
        {"final/modulation",
         timesteps * h3_final_modulation_rows * f.hidden * sizeof(float)}
    );
    m.tensors.push_back({"final/time-ids", tokens * sizeof(std::uint32_t)});
    if (video)
        m.tensors.push_back({"final/video", video * f.video_width * sizeof(float)});
    if (audio)
        m.tensors.push_back({"final/audio", audio * f.audio_width * sizeof(float)});
    m.retained_inputs = {"resident/hidden", "block/time"};
    m.workspaces_complete = true;
    return m;
}

MediaShape h3_final_shape(const H3Final& f, std::size_t tokens) {
    // The planner's FF tiles go unused; size them at the hidden width.
    return {tokens, f.hidden, 0, 0, 0, f.hidden};
}

// Every device holds every refiner weight and computes the same text rows on its
// own hidden copy, so no peer gathers are needed.
void execute_h3_refiner(
    MediaSession& session,
    std::span<H3ResidentLauncher* const> launchers,
    const H3Refiner& r,
    std::size_t tokens,
    std::size_t text,
    const std::atomic_bool& cancel
) {
    const auto& plan = session.plan();
    const auto inner = r.heads * r.head_dim;
    require(
        plan.schedule == MediaSchedule::feed_forward && plan.rows && text
            && text <= tokens,
        "h3_resident_plan"
    );
    auto all = lanes(session, launchers);
    for (auto& lane : all) {
        lane.qkv = bind(session, lane.id, r.qkv, 3, inner, r.hidden);
        lane.out = bind(session, lane.id, r.out, 1, r.hidden, inner);
        lane.fc1 = bind(session, lane.id, r.fc1, 2, r.width, r.hidden);
        lane.fc2 = bind(session, lane.id, r.fc2, 1, r.hidden, r.width);
        for (std::size_t n = 0; n < r.norm.size(); ++n)
            lane.norm[n] = norm_view(session, lane.id, r.norm[n]);
        lane.hidden = matrix(session, lane.id, "resident/hidden", tokens, r.hidden);
        lane.normalized =
            matrix(session, lane.id, "resident/normalized", tokens, r.hidden);
        std::vector<DeviceTensorView*> views{&lane.hidden};
        for (auto& norm : lane.norm)
            views.push_back(&norm);
        if (!r.final_norm.empty()) {
            lane.final_norm = norm_view(session, lane.id, r.final_norm);
            views.push_back(&lane.final_norm);
        }
        for (auto* p : {&lane.qkv, &lane.out, &lane.fc1, &lane.fc2})
            ready(*p, views);
        session.require_ready(views);
    }
    session.advance(MediaPhase::ff);
    const auto queries = std::min<std::size_t>({128, plan.rows, text}),
               key_tile = std::min<std::size_t>(4096, text);
    auto tiles = [&](std::size_t size, auto body) {
        for (std::size_t first = 0; first < text;) {
            stop(cancel);
            const auto rows = std::min(size, text - first);
            body(first, rows);
            first += rows;
        }
    };
    for (auto& lane : all) {
        bind_scratch(session, lane);
        auto query = matrix(session, lane.id, "refiner/query", text, inner);
        auto key = matrix(session, lane.id, "refiner/key", text, inner);
        auto value = matrix(session, lane.id, "refiner/value", text, inner);
        // Q, K and V for every text token before any hidden row changes.
        tiles(plan.rows, [&](std::size_t first, std::size_t rows) {
            auto normalized = session.rows(lane.normalized, first, rows);
            normalize(session, lane, lane.norm[0], first, rows, normalized);
            auto q = session.rows(query, first, rows);
            project(session, lane, lane.qkv, 0, normalized, q, rows);
            normalize_qk(session, lane, q, 2, first);
            auto k = session.rows(key, first, rows);
            project(session, lane, lane.qkv, 1, normalized, k, rows);
            normalize_qk(session, lane, k, 3, first);
            auto v = session.rows(value, first, rows);
            project(session, lane, lane.qkv, 2, normalized, v, rows);
        });
        tiles(queries, [&](std::size_t first, std::size_t rows) {
            auto q = session.rows(query, first, rows);
            auto attended = matrix(session, lane.id, "refiner/attended", rows, inner);
            attend(session, lane, q, key, value, attended, r.heads, rows, key_tile);
            auto projected =
                matrix(session, lane.id, "resident/residual-tile", rows, r.hidden);
            project(session, lane, lane.out, 0, attended, projected, rows);
            add(session, lane, projected, first, rows);
        });
        tiles(plan.rows, [&](std::size_t first, std::size_t rows) {
            auto normalized = session.rows(lane.normalized, first, rows);
            normalize(session, lane, lane.norm[1], first, rows, normalized);
            auto pair =
                matrix(session, lane.id, "resident/ff-gate-up", 2 * rows, r.width);
            auto gate = session.rows(pair, 0, rows);
            auto up = session.rows(pair, rows, rows);
            project(session, lane, lane.fc1, 0, normalized, gate, rows);
            project(session, lane, lane.fc1, 1, normalized, up, rows);
            const std::array read{&gate, &up};
            const std::array write{&gate, &lane.status};
            run(session, lane, read, write, [&](void* stream, auto in, auto out) {
                lane.launcher->silu_gate(stream, in, out);
            });
            auto down =
                matrix(session, lane.id, "resident/residual-tile", rows, r.hidden);
            project(session, lane, lane.fc2, 0, gate, down, rows);
            add(session, lane, down, first, rows);
        });
        if (!r.final_norm.empty())
            tiles(plan.rows, [&](std::size_t first, std::size_t rows) {
                normalize(session, lane, lane.final_norm, first, rows, lane.hidden);
            });
    }
    all.clear();
    session.advance(MediaPhase::complete);
}

void execute_h3_final(
    MediaSession& session,
    std::span<H3ResidentLauncher* const> launchers,
    const H3Final& f,
    std::size_t tokens,
    std::size_t text,
    std::span<float> video,
    std::span<float> audio,
    const std::atomic_bool& cancel
) {
    const auto& plan = session.plan();
    const auto nv = video.size() / f.video_width, na = audio.size() / f.audio_width;
    require(
        plan.schedule == MediaSchedule::feed_forward && plan.rows
            && video.size() == nv * f.video_width && audio.size() == na * f.audio_width
            && (nv || na) && text + nv + na == tokens,
        "h3_resident_plan"
    );
    const auto& tensors = plan.tensors;
    const auto time = std::find_if(tensors.begin(), tensors.end(), [](const auto& t) {
        return t.name == "block/time";
    });
    require(time != tensors.end(), "h3_resident_plan");
    const auto timesteps = std::size_t(time->bytes / sizeof(float) / f.time_width);
    auto all = lanes(session, launchers);
    std::vector<DeviceTensorView> outputs[2];
    for (auto& lane : all) {
        lane.adaln = bind(
            session,
            lane.id,
            f.adaln,
            1,
            h3_final_modulation_rows * f.hidden,
            f.time_width
        );
        if (nv)
            lane.video = bind(session, lane.id, f.video, 1, f.video_width, f.hidden);
        if (na)
            lane.audio = bind(session, lane.id, f.audio, 1, f.audio_width, f.hidden);
        lane.final_norm = norm_view(session, lane.id, f.norm);
        lane.hidden = matrix(session, lane.id, "resident/hidden", tokens, f.hidden);
        lane.normalized =
            matrix(session, lane.id, "resident/normalized", tokens, f.hidden);
        lane.time = matrix(session, lane.id, "block/time", timesteps, f.time_width);
        lane.modulation = matrix(
            session,
            lane.id,
            "final/modulation",
            timesteps,
            h3_final_modulation_rows * f.hidden
        );
        lane.time_ids =
            session.tensor(lane.id, "final/time-ids", DeviceDType::u32, {tokens}, {1});
        std::vector<DeviceTensorView*> views{
            &lane.hidden, &lane.time, &lane.time_ids, &lane.final_norm
        };
        ready(lane.adaln, views);
        if (nv)
            ready(lane.video, views);
        if (na)
            ready(lane.audio, views);
        session.require_ready(views);
    }
    session.advance(MediaPhase::ff);
    for (auto& lane : all) {
        bind_scratch(session, lane);
        project(session, lane, lane.adaln, 0, lane.time, lane.modulation, timesteps);

        // Only the selected video then audio rows are normalized and projected.
        struct OutputSegment {
            std::size_t start, count;
            Projection* projection;
            const char* name;
            std::size_t width;
        };

        const std::array<OutputSegment, 2> segments{
            {{text, nv, &lane.video, "final/video", f.video_width},
             {text + nv, na, &lane.audio, "final/audio", f.audio_width}}
        };
        std::size_t index = 0;
        for (const auto& [start, count, projection, name, width] : segments) {
            auto& output = outputs[index++];
            if (!count) {
                output.emplace_back();
                continue;
            }
            output.push_back(matrix(session, lane.id, name, count, width));
            for (std::size_t first = start; first < start + count;) {
                stop(cancel);
                const auto rows = std::min(plan.rows, start + count - first);
                auto normalized = session.rows(lane.normalized, first, rows);
                normalize(
                    session,
                    lane,
                    lane.final_norm,
                    first,
                    rows,
                    normalized,
                    &lane.modulation,
                    &lane.time_ids,
                    0,
                    h3_final_modulation_rows
                );
                auto out = session.rows(output.back(), first - start, rows);
                project(session, lane, *projection, 0, normalized, out, rows);
                first += rows;
            }
        }
    }
    // Every device computes the same rows; download the first copy.
    auto video_view = std::move(outputs[0].front());
    auto audio_view = std::move(outputs[1].front());
    for (auto& output : outputs)
        output.clear();
    all.clear();
    session.advance(MediaPhase::complete);
    if (nv)
        session.download_output(video_view, std::as_writable_bytes(video));
    if (na)
        session.download_output(audio_view, std::as_writable_bytes(audio));
}
}  // namespace kadan::video::detail
