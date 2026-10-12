#include "kadan/image.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "image_activation.hpp"
#include "kadan/bf16_projection.hpp"
#include "kadan/checkpoint.hpp"
#include "kadan/checkpoint_floats.hpp"
#include "kadan/float_weights.hpp"
#ifdef KADAN_IMAGE_BLAS
#include <cblas.h>
#endif
namespace kadan::image {
namespace {
void need(bool b, const char* e) {
    if (!b)
        throw std::runtime_error(e);
}

void stop(const std::atomic_bool& c) {
    need(!c.load(), "image_cancelled");
}

Footprint host(Resources& r, Bytes n) {
    auto p = r.snapshot().capacity;
    std::fill(p.begin(), p.end(), 0);
    p[0] = n;
    return p;
}

struct Lease {
    Resources& r;
    Handle h;

    Lease(Resources& a, Bytes n) : r(a), h(r.reserve(Workload::image, host(r, n))) {
    }

    ~Lease() {
        r.released(h);
    }
};

// Every consumed element is first written by checkpoint decoding, input copy,
// normalization, residual or a complete projection/attention output. Tails are unused.
struct Buffer {
    Lease lease;
    std::unique_ptr<float[]> data;
    std::size_t n;

    Buffer(Resources& r, std::size_t count)
        : lease(r, count * 4), data(std::make_unique_for_overwrite<float[]>(count)),
          n(count) {
    }

    std::span<float> span() {
        return {data.get(), n};
    }
};

struct Busy {
    bool& b;

    Busy(bool& flag) : b(flag) {
        need(!b, "busy");
        b = true;
    }

    ~Busy() {
        b = false;
    }
};

struct Pin {
    Resources& r;
    Handle h;

    Pin(Resources& a, Handle id) : r(a), h(id) {
        r.pin(h);
    }

    ~Pin() {
        r.unpin(h);
    }
};

void finite(
    std::span<const float> x, ProjectionResult result = ProjectionResult::unvalidated
) {
    need(projection_output_finite(result, x), "image_nonfinite");
}

struct Spec {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::size_t offset = 0, count = 1, shard = 0;
    WeightIdentity identity{};
};

std::vector<Spec> layout(BlockConfig d) {
    std::vector<Spec> out;
    auto add = [&](std::string p, std::initializer_list<std::uint64_t> shape) {
        out.push_back({std::move(p), shape});
    };
    for (auto n : {"to_q", "to_k", "to_v", "to_out.0"})
        add(std::string("attn.") + n + ".weight", {d.state, d.state});
    for (auto n : {"norm_q", "norm_k"})
        add(std::string("attn.") + n + ".weight", {d.head_dim});
    for (auto n : {"proj", "gate_layer"}) {
        add(std::string("img_mlp.") + n + ".weight", {d.intermediate, d.state});
    }
    add("img_mlp.out.weight", {d.state, d.intermediate});
    std::size_t offset = 0;
    for (auto& s : out) {
        for (auto n : s.shape)
            s.count *= n;
        s.offset = offset;
        offset += s.count;
    }
    return out;
}

void validate(BlockConfig d) {
    need(
        d.state > 0 && d.state <= 4096 && d.heads > 0 && d.heads <= 32 && d.head_dim > 0
            && d.head_dim <= 128 && d.state == d.heads * d.head_dim
            && d.intermediate > 0 && d.intermediate <= 12288,
        "image_config"
    );
    std::size_t sum = 0;
    for (auto a : d.axes) {
        need(a % 2 == 0, "image_rope_config");
        sum += a;
    }
    need(sum == d.head_dim, "image_rope_config");
}
}  // namespace

struct TransformerBlock::Impl {
    Resources& r;
    std::shared_ptr<DenseCompute> compute;
    std::unique_ptr<Lease> parser;
    std::vector<std::unique_ptr<checkpoint::Shard>> shards;
    std::string prefix;
    Lease metadata;
    BlockConfig d;
    std::vector<Spec> specs;
    std::map<std::string, std::size_t, std::less<>> names;
    std::unique_ptr<Buffer> weights;

    Impl(Resources& r, BlockConfig config, std::shared_ptr<DenseCompute> c)
        : r(r), compute(std::move(c)), metadata(r, 1024 * 1024), d(config),
          specs(layout(config)) {
        for (std::size_t i = 0; i < specs.size(); ++i)
            names.emplace(specs[i].name, i);
    }

    checkpoint::FloatSlice
    weight(const std::string& n, const std::atomic_bool& cancel) {
        const auto& s = specs.at(names.at(n));
        return checkpoint::float_slice(
            r, Workload::image, *shards[s.shard], prefix + n, 0, s.count, cancel
        );
    }

    void linear(
        const std::string& name,
        std::span<const float> x,
        std::span<float> y,
        std::size_t rows,
        std::size_t in,
        std::size_t out,
        const std::atomic_bool& cancel,
        ProjectionPolicy policy = ProjectionPolicy::standard
    ) {
        const auto& spec = specs.at(names.at(name + ".weight"));
        need(
            spec.count == in * out && x.size() == rows * in && y.size() == rows * out,
            "image_projection_shape"
        );
        const auto result = checkpoint::project_source(
            r,
            Workload::image,
            compute.get(),
            *shards[spec.shard],
            prefix + name + ".weight",
            {},
            x,
            in,
            out,
            y,
            cancel,
            policy
        );
        finite(y, result);
    }

    bool silu_pair(
        std::span<const float> input,
        std::span<float> output,
        const std::atomic_bool& cancel
    ) {
        if (!compute || !compute->supports_bf16_weights())
            return false;
        const auto& gate = specs.at(names.at("img_mlp.gate_layer.weight"));
        const auto& up = specs.at(names.at("img_mlp.proj.weight"));
        auto& gate_shard = *shards[gate.shard];
        auto& up_shard = *shards[up.shard];
        const auto gate_name = prefix + gate.name, up_name = prefix + up.name;
        if (gate_shard.tensor(gate_name).dtype != checkpoint::Dtype::bf16
            || up_shard.tensor(up_name).dtype != checkpoint::Dtype::bf16)
            return false;
        auto source = [&](checkpoint::Shard& shard, const std::string& name) {
            return [&shard,
                    &name,
                    &cancel](std::size_t first, std::span<std::uint16_t> dst) {
                checkpoint::read_bf16(
                    first,
                    dst,
                    cancel,
                    [&](std::size_t offset, std::span<std::uint8_t> bytes) {
                        shard.read_tensor(name, offset, bytes);
                    }
                );
            };
        };
        gate_shard.check_unchanged();
        up_shard.check_unchanged();
        const bool completed = compute->image_silu_pair_bf16_source(
            gate.identity,
            source(gate_shard, gate_name),
            up.identity,
            source(up_shard, up_name),
            input,
            d.state,
            d.intermediate,
            output,
            cancel
        );
        gate_shard.check_unchanged();
        up_shard.check_unchanged();
        return completed;
    }

    void norm_mod(
        std::span<const float> x,
        std::span<const float> modulation,
        std::span<float> y,
        std::size_t rows,
        std::size_t text,
        std::size_t stage,
        const std::atomic_bool& cancel
    ) {
        for (std::size_t t = 0; t < rows; ++t) {
            stop(cancel);
            double mean = 0, variance = 0;
            for (std::size_t c = 0; c < d.state; ++c)
                mean += x[t * d.state + c];
            mean /= d.state;
            for (std::size_t c = 0; c < d.state; ++c) {
                double v = x[t * d.state + c] - mean;
                variance += v * v;
            }
            float scale = float(1 / std::sqrt(variance / d.state + 1e-6));
            auto offset = (t < text ? 4 * d.state : 0) + stage * 2 * d.state;
            for (std::size_t c = 0; c < d.state; ++c)
                y[t * d.state + c] = float(x[t * d.state + c] - mean) * scale
                                     * (1 + modulation[offset + c]);
        }
        finite(y);
    }

    void rotate(
        std::span<float> x,
        const std::string& name,
        std::size_t rows,
        std::size_t text,
        std::size_t height,
        std::size_t width,
        const std::atomic_bool& cancel
    ) {
        auto norm = weight(name + ".weight", cancel);
        for (std::size_t t = 0; t < rows; ++t) {
            stop(cancel);
            std::array<long, 3> positions{long(t), long(t), long(t)};
            if (t >= text)
                positions = {
                    long(text),
                    long((t - text) / width) - long(height - height / 2),
                    long((t - text) % width) - long(width - width / 2)
                };
            // Every head at this position uses the same rotary coefficients.
            std::array<float, 64> cosine{}, sine{};
            std::size_t offset = 0;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                for (std::size_t c = 0; c < d.axes[axis]; c += 2) {
                    float angle = float(positions[axis])
                                  * std::pow(10000.f, -float(c) / float(d.axes[axis]));
                    cosine[(offset + c) / 2] = std::cos(angle);
                    sine[(offset + c) / 2] = std::sin(angle);
                }
                offset += d.axes[axis];
            }
            for (std::size_t h = 0; h < d.heads; ++h) {
                auto row = x.subspan(t * d.state + h * d.head_dim, d.head_dim);
                double variance = 0;
                for (float v : row)
                    variance += double(v) * v;
                float scale = float(1 / std::sqrt(variance / d.head_dim + 1e-6));
                for (std::size_t c = 0; c < d.head_dim; ++c)
                    row[c] *= scale * norm[c];
                for (std::size_t c = 0; c < d.head_dim; c += 2) {
                    float co = cosine[c / 2], si = sine[c / 2], a = row[c],
                          b = row[c + 1];
                    row[c] = a * co - b * si;
                    row[c + 1] = b * co + a * si;
                }
            }
        }
        finite(x);
    }
};

TransformerBlock::TransformerBlock(
    std::shared_ptr<Resources> r, std::shared_ptr<DenseCompute> compute
)
    : resources_(std::move(r)), compute_(std::move(compute)) {
    need(bool(resources_), "image_resources");
}

TransformerBlock::~TransformerBlock() {
    unload();
}

std::optional<MediaPlanError> TransformerBlock::prepare_feed_forward(
    MediaSession& session, std::size_t tokens, const std::atomic_bool& cancel
) {
    Busy active(busy_);
    stop(cancel);
    need(bool(model_), "image_load_state");
    auto& m = *model_;
    StageManifest manifest;
    manifest.schedule = MediaSchedule::feed_forward;
    manifest.blas_bytes = video::bf16_projection_workspace_bytes;
    manifest.status_bytes = sizeof(std::uint32_t);  // Aggregate certificate.
    manifest.tensors = {
        {"ff/residual-gates", 2 * m.d.state * sizeof(float)},
        {"ff/gate-status",
         sizeof(std::uint32_t),
         MediaPhase::ff,
         MediaPhase::ff,
         AllocationKind::status},
        {"ff/up-status",
         sizeof(std::uint32_t),
         MediaPhase::ff,
         MediaPhase::ff,
         AllocationKind::status}
    };
    for (const auto* name :
         {"img_mlp.gate_layer.weight", "img_mlp.proj.weight", "img_mlp.out.weight"}) {
        const auto& spec = m.specs.at(m.names.at(name));
        auto& shard = *m.shards[spec.shard];
        shard.check_unchanged();
        const auto qualified = m.prefix + spec.name;
        const auto tensor = shard.tensor(qualified);
        if (tensor.dtype != checkpoint::Dtype::bf16)
            return MediaPlanError{PlanFailure::unsupported_layout, "image_ff_bf16", {}};
        need(
            shard.tensor_identity(qualified) == spec.identity, "image_weight_identity"
        );
        manifest.weights.push_back(
            {qualified,
             spec.identity,
             TensorStorage::bf16_exact,
             std::size_t(tensor.shape[0]),
             std::size_t(tensor.shape[1]),
             WeightPartition::channels,
             1,
             1,
             ProjectionPrecision::bf16_f32}
        );
    }
    // Planner owns full hidden/normalized boundaries, combined gate/up output,
    // ordered full-width FF gather, down/residual tile, and reusable BF16 input.
    // No normalization/attention/LoRA operator is part of this FF boundary.
    manifest.workspaces_complete = true;
    if (auto error = session.admit(
            manifest, MediaShape{tokens, m.d.state, 0, 0, 0, m.d.intermediate}
        ))
        return error;
    try {
        for (const auto& slice : session.plan().weights) {
            stop(cancel);
            const auto& spec =
                m.specs.at(m.names.at(slice.name.substr(m.prefix.size())));
            auto& shard = *m.shards[spec.shard];
            shard.check_unchanged();
            Lease staging(*resources_, slice.stored_bytes);
            std::vector<std::uint8_t> raw(slice.stored_bytes);
            shard.read_tensor(slice.name, slice.first_row * spec.shape[1] * 2, raw);
            shard.check_unchanged();
            stop(cancel);
            auto destination = session.weight(slice);
            session.upload_weight(destination, std::as_bytes(std::span(raw)));
        }
    } catch (...) {
        session.abort(std::current_exception());
        throw;
    }
    return {};
}

void TransformerBlock::execute_feed_forward(
    MediaSession& session,
    std::size_t tokens,
    std::size_t text,
    std::span<video::ImageFeedForwardLauncher* const> launchers,
    const std::atomic_bool& cancel
) {
    Busy active(busy_);
    stop(cancel);
    need(bool(model_), "image_load_state");
    auto& m = *model_;
    const auto& plan = session.plan();
    need(
        tokens && text <= tokens && plan.rows && plan.rows <= tokens
            && plan.schedule == MediaSchedule::feed_forward
            && launchers.size() == plan.devices.size()
            && plan.weights.size() == 3 * plan.devices.size(),
        "image_ff_plan"
    );

    struct Device {
        video::ImageFeedForwardLauncher* launcher;
        const WeightSlice *gate_slice, *up_slice, *down_slice;
        DeviceTensorView gate_weight, up_weight, down_weight, normalized, hidden, gates;
    };

    try {
        std::vector<Device> devices;
        for (const auto& planned : plan.devices) {
            video::ImageFeedForwardLauncher* launcher = nullptr;
            for (auto* candidate : launchers) {
                need(candidate, "image_ff_launcher");
                if (candidate->device() == planned.device) {
                    need(!launcher, "image_ff_launcher");
                    launcher = candidate;
                }
            }
            need(launcher, "image_ff_launcher");
            auto slice = [&](const char* name) -> const WeightSlice& {
                const auto& spec = m.specs.at(m.names.at(name));
                const auto found = std::find_if(
                    plan.weights.begin(), plan.weights.end(), [&](const auto& w) {
                        return w.device == planned.device && w.name == m.prefix + name
                               && w.identity == spec.identity;
                    }
                );
                need(found != plan.weights.end(), "image_ff_weight");
                return *found;
            };
            const auto& gate = slice("img_mlp.gate_layer.weight");
            const auto& up = slice("img_mlp.proj.weight");
            const auto& down = slice("img_mlp.out.weight");
            need(
                gate.first_row == up.first_row && gate.rows == up.rows,
                "image_ff_pair_partition"
            );
            auto matrix = [&](const char* name, std::size_t rows, std::size_t columns) {
                return session.tensor(
                    planned.device,
                    name,
                    DeviceDType::f32,
                    {rows, columns},
                    {columns, 1}
                );
            };
            devices.push_back(
                {launcher,
                 &gate,
                 &up,
                 &down,
                 session.weight(gate),
                 session.weight(up),
                 session.weight(down),
                 matrix("resident/normalized", tokens, m.d.state),
                 matrix("resident/hidden", tokens, m.d.state),
                 matrix("ff/residual-gates", 2, m.d.state)}
            );
            auto& d = devices.back();
            const std::array ready{
                &d.normalized,
                &d.hidden,
                &d.gates,
                &d.gate_weight,
                &d.up_weight,
                &d.down_weight
            };
            session.require_ready(ready);
        }
        // Gate/up and down have independently partitioned output channel domains.
        auto ordered = [&](bool down) {
            std::vector<Device*> result;
            result.reserve(devices.size());
            for (auto& d : devices)
                result.push_back(&d);
            std::sort(result.begin(), result.end(), [&](auto* a, auto* b) {
                return (down ? a->down_slice : a->gate_slice)->first_row
                       < (down ? b->down_slice : b->gate_slice)->first_row;
            });
            std::size_t column = 0;
            for (auto* d : result) {
                const auto& w = *(down ? d->down_slice : d->gate_slice);
                need(
                    w.first_row == column && w.rows
                        && w.rows <= (down ? m.d.state : m.d.intermediate) - column,
                    "image_ff_partition"
                );
                column += w.rows;
            }
            need(column == (down ? m.d.state : m.d.intermediate), "image_ff_partition");
            return result;
        };
        const auto gate_order = ordered(false), down_order = ordered(true);
        session.advance(MediaPhase::ff);
        for (std::size_t first = 0; first < tokens;) {
            stop(cancel);
            const auto rows = std::min(plan.rows, tokens - first);

            struct Tile {
                Device* device;
                DeviceTensorView input, hidden, pair, activated, rounded, workspace,
                    gate_status, up_status, status, ff_input, down, gathered;
            };

            std::vector<Tile> tiles;
            for (auto& d : devices) {
                const auto device = d.launcher->device();
                auto matrix = [&](const char* name,
                                  DeviceDType type,
                                  std::size_t r,
                                  std::size_t c) {
                    return session.tensor(device, name, type, {r, c}, {c, 1});
                };
                auto status = [&](const char* name) {
                    return session.tensor(device, name, DeviceDType::u32, {1}, {1});
                };
                Tile t;
                t.device = &d;
                t.input = session.rows(d.normalized, first, rows);
                t.hidden = session.rows(d.hidden, first, rows);
                t.pair = matrix(
                    "resident/ff-gate-up",
                    DeviceDType::f32,
                    2 * rows,
                    d.gate_slice->rows
                );
                t.rounded = matrix(
                    "resident/projection-conversion", DeviceDType::bf16, rows, m.d.state
                );
                t.workspace = session.tensor(
                    device,
                    "resident/blas",
                    DeviceDType::u8,
                    {video::bf16_projection_workspace_bytes},
                    {1}
                );
                t.gate_status = status("ff/gate-status");
                t.up_status = status("ff/up-status");
                t.status = status("resident/status");
                t.ff_input = matrix(
                    "resident/ff-input", DeviceDType::f32, rows, m.d.intermediate
                );
                t.down = matrix(
                    "resident/residual-tile", DeviceDType::f32, rows, d.down_slice->rows
                );
                const std::array inputs{&t.input, &d.gate_weight, &d.up_weight};
                const std::array outputs{
                    &t.pair,
                    &t.rounded,
                    &t.workspace,
                    &t.gate_status,
                    &t.up_status,
                    &t.status
                };
                session.launch(
                    device, inputs, outputs, [&](void* stream, auto read, auto write) {
                        d.launcher->pair(stream, read, write);
                    }
                );
                session.complete(outputs, t.status);
                t.activated = session.rows(t.pair, 0, rows);
                tiles.push_back(std::move(t));
            }
            auto tile = [&](Device* d) -> Tile& {
                return *std::find_if(tiles.begin(), tiles.end(), [&](const auto& t) {
                    return t.device == d;
                });
            };
            for (auto& t : tiles) {
                for (auto* d : gate_order)
                    session.gather(
                        tile(d).activated, t.ff_input, d->gate_slice->first_row
                    );
                // Release the old geometry before rebinding the reusable allocation.
                t.rounded = {};
                const auto device = t.device->launcher->device();
                t.rounded = session.tensor(
                    device,
                    "resident/projection-conversion",
                    DeviceDType::bf16,
                    {rows, m.d.intermediate},
                    {m.d.intermediate, 1}
                );
                const std::array inputs{&t.ff_input, &t.device->down_weight};
                const std::array outputs{&t.down, &t.rounded, &t.workspace, &t.status};
                session.launch(
                    device, inputs, outputs, [&](void* stream, auto read, auto write) {
                        t.device->launcher->down(stream, read, write);
                    }
                );
                session.complete(outputs, t.status);
            }
            for (auto& t : tiles) {
                auto* projected = &t.down;
                const auto device = t.device->launcher->device();
                if (tiles.size() > 1) {
                    t.gathered = session.tensor(
                        device,
                        "resident/peer-tile",
                        DeviceDType::f32,
                        {rows, m.d.state},
                        {m.d.state, 1}
                    );
                    for (auto* d : down_order)
                        session.gather(
                            tile(d).down, t.gathered, d->down_slice->first_row
                        );
                    projected = &t.gathered;
                }
                const std::array inputs{&t.hidden, projected, &t.device->gates};
                const std::array outputs{&t.hidden, &t.status};
                session.launch(
                    device, inputs, outputs, [&](void* stream, auto read, auto write) {
                        t.device->launcher->residual(
                            stream, write[0], read[1], read[2], write[1], first, text
                        );
                    }
                );
                session.complete(outputs, t.status);
            }
            first += rows;
        }
    } catch (...) {
        session.abort(std::current_exception());
        throw;
    }
    session.advance(MediaPhase::complete);
}

void TransformerBlock::load(
    const char* root,
    const std::string& filename,
    std::size_t block,
    BlockConfig d,
    const std::atomic_bool& cancel
) {
    const std::array<std::string, 1> files{filename};
    load(root, files, block, d, cancel);
}

void TransformerBlock::load(
    const char* root,
    std::span<const std::string> files,
    std::size_t block,
    BlockConfig d,
    const std::atomic_bool& cancel
) {
    Busy active(busy_);
    stop(cancel);
    need(!model_ && block < 32, "image_load_state");
    validate(d);
    need(
        !files.empty() && files.size() <= 4
            && std::set<std::string>(files.begin(), files.end()).size() == files.size(),
        "image_shards"
    );
    auto m = std::make_unique<Impl>(*resources_, d, compute_);
    m->parser = std::make_unique<Lease>(*resources_, files.size() * 16 * 1024 * 1024);
    auto& shards = m->shards;
    for (const auto& filename : files) {
        shards.push_back(
            std::make_unique<checkpoint::Shard>(
                root,
                filename,
                std::make_shared<checkpoint::MemoryBudget>(16 * 1024 * 1024),
                checkpoint::Limits{2 * 1024 * 1024, 4096, 4096}
            )
        );
    }
    auto prefix = "transformer_blocks." + std::to_string(block) + ".";
    m->prefix = prefix;
    for (auto& spec : m->specs) {
        std::size_t found = shards.size();
        for (std::size_t i = 0; i < shards.size(); ++i) {
            try {
                shards[i]->tensor(prefix + spec.name);
            } catch (const std::invalid_argument& e) {
                if (std::string(e.what()) == "missing_tensor")
                    continue;
                throw;
            }
            need(found == shards.size(), "image_duplicate_tensor");
            found = i;
        }
        need(found < shards.size(), "image_missing_tensor");
        spec.shard = found;
        spec.identity = shards[found]->tensor_identity(prefix + spec.name);
        auto t = shards[found]->tensor(prefix + spec.name);
        need(
            (t.dtype == checkpoint::Dtype::bf16 || t.dtype == checkpoint::Dtype::fp32)
                && t.rank == spec.shape.size()
                && std::equal(spec.shape.begin(), spec.shape.end(), t.shape.begin()),
            "image_tensor_layout"
        );
    }
    m->weights = std::make_unique<Buffer>(*resources_, 1);
    for (const auto& shard : shards) {
        shard->check_unchanged();
    }
    stop(cancel);
    resources_->loaded(m->weights->lease.h);
    model_ = std::move(m);
}

void TransformerBlock::unload() {
    need(!busy_, "busy");
    if (model_) {
        resources_->begin_eviction(model_->weights->lease.h);
        model_.reset();
    }
}

void TransformerBlock::execute(
    std::span<const float> input,
    std::span<const float> modulation,
    std::size_t text,
    std::size_t height,
    std::size_t width,
    std::span<float> output,
    const std::atomic_bool& cancel,
    const Hook& hook
) {
    Busy active(busy_);
    stop(cancel);
    need(bool(model_), "image_not_loaded");
    auto& m = *model_;
    auto d = m.d;
    need(
        text > 0 && text <= 2048 && height > 0 && height <= 256 && width > 0
            && width <= 256 && text + height * width <= 32768,
        "image_token_layout"
    );
    const auto rows = text + height * width;
    need(
        input.size() == rows * d.state && output.size() == input.size()
            && modulation.size() == 8 * d.state,
        "image_shape"
    );
    finite(input);
    finite(modulation);
    Pin pin(*resources_, m.weights->lease.h);
    Buffer gates(*resources_, 4 * d.state), x(*resources_, input.size()),
        n(*resources_, input.size()), q(*resources_, input.size()),
        k(*resources_, input.size()), v(*resources_, input.size()),
        mix(*resources_, input.size()), delta(*resources_, input.size()),
        scores(*resources_, rows), gate(*resources_, rows * d.intermediate);
    // Gates depend on text/image and channel, not on the token position.
    for (std::size_t group = 0; group < 2; ++group)
        for (std::size_t stage = 0; stage < 2; ++stage)
            for (std::size_t c = 0; c < d.state; ++c)
                gates.data[(group * 2 + stage) * d.state + c] =
                    std::tanh(modulation[(group * 4 + stage * 2 + 1) * d.state + c]);
    m.norm_mod(input, modulation, n.span(), rows, text, 0, cancel);
    for (auto item :
         {std::pair{"attn.to_q", q.span()},
          std::pair{"attn.to_k", k.span()},
          std::pair{"attn.to_v", v.span()}})
        m.linear(item.first, n.span(), item.second, rows, d.state, d.state, cancel);
    m.rotate(q.span(), "attn.norm_q", rows, text, height, width, cancel);
    m.rotate(k.span(), "attn.norm_k", rows, text, height, width, cancel);
    if (hook)
        hook("qkv");
    stop(cancel);
    if (!(m.compute
          && m.compute->attend(
              q.span(),
              k.span(),
              v.span(),
              rows,
              rows,
              d.heads,
              d.heads,
              d.head_dim,
              text,
              0,
              0,
              mix.span(),
              cancel
          )))
        for (std::size_t t = 0; t < rows; ++t) {
            const auto allowed = t < text ? t + 1 : rows;
            for (std::size_t h = 0; h < d.heads; ++h) {
                stop(cancel);
                float maximum = -INFINITY;
                for (std::size_t s = 0; s < allowed; ++s) {
                    float value = 0;
                    for (std::size_t c = 0; c < d.head_dim; ++c)
                        value += q.data[t * d.state + h * d.head_dim + c]
                                 * k.data[s * d.state + h * d.head_dim + c];
                    scores.data[s] = value / std::sqrt(float(d.head_dim));
                    maximum = std::max(maximum, scores.data[s]);
                }
                double sum = 0;
                for (std::size_t s = 0; s < allowed; ++s) {
                    scores.data[s] = std::exp(scores.data[s] - maximum);
                    sum += scores.data[s];
                }
                need(sum > 0 && std::isfinite(sum), "image_nonfinite");
                for (std::size_t c = 0; c < d.head_dim; ++c) {
                    float value = 0;
                    for (std::size_t s = 0; s < allowed; ++s)
                        value += float(scores.data[s] / sum)
                                 * v.data[s * d.state + h * d.head_dim + c];
                    mix.data[t * d.state + h * d.head_dim + c] = value;
                }
            }
        }
    m.linear("attn.to_out.0", mix.span(), delta.span(), rows, d.state, d.state, cancel);
    for (std::size_t t = 0; t < rows; ++t)
        for (std::size_t c = 0; c < d.state; ++c)
            x.data[t * d.state + c] = input[t * d.state + c]
                                      + gates.data[(t < text ? 2 * d.state : 0) + c]
                                            * delta.data[t * d.state + c];
    if (hook)
        hook("attention");
    m.norm_mod(x.span(), modulation, n.span(), rows, text, 1, cancel);
    if (!m.silu_pair(n.span(), gate.span(), cancel)) {
        Buffer up(*resources_, rows * d.intermediate);
        m.linear(
            "img_mlp.gate_layer",
            n.span(),
            gate.span(),
            rows,
            d.state,
            d.intermediate,
            cancel,
            ProjectionPolicy::image_feed_forward
        );
        m.linear(
            "img_mlp.proj",
            n.span(),
            up.span(),
            rows,
            d.state,
            d.intermediate,
            cancel,
            ProjectionPolicy::image_feed_forward
        );
        for (std::size_t i = 0; i < gate.n; ++i)
            gate.data[i] = detail::silu_product(gate.data[i], up.data[i]);
    }
    m.linear(
        "img_mlp.out",
        gate.span(),
        delta.span(),
        rows,
        d.intermediate,
        d.state,
        cancel,
        ProjectionPolicy::image_feed_forward
    );
    for (std::size_t t = 0; t < rows; ++t)
        for (std::size_t c = 0; c < d.state; ++c)
            x.data[t * d.state + c] +=
                gates.data[(t < text ? 2 * d.state : 0) + d.state + c]
                * delta.data[t * d.state + c];
    finite(x.span());
    if (hook)
        hook("feed_forward");
    stop(cancel);
    std::copy(x.span().begin(), x.span().end(), output.begin());
}
}  // namespace kadan::image
