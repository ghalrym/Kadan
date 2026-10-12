#include "kadan/image_denoiser.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <set>

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
// normalization or a complete projection/attention output. Scratch tails are unused.
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

void finite(std::span<const float> x) {
    for (float v : x)
        need(std::isfinite(v), "image_nonfinite");
}

struct Spec {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::size_t offset = 0, count = 1, shard = 0;
    WeightIdentity identity{};
};

std::vector<Spec> layout(DenoiserConfig d) {
    std::vector<Spec> out;
    auto add = [&](std::string p, std::initializer_list<std::uint64_t> shape) {
        out.push_back({std::move(p), shape});
    };
    const auto n = d.block.state;
    add("img_in.weight", {n, d.channels});
    add("txt_in.text_norm.weight", {d.context});
    add("txt_in.in_layer.weight", {n, d.context});
    add("txt_in.out_layer.weight", {n, n});
    add("time_text_embed.timestep_embedder.linear_1.weight", {n, 256});
    add("time_text_embed.timestep_embedder.linear_2.weight", {n, n});
    add("modulation.1.weight", {4 * n, n});
    add("norm_out.linear.weight", {n, n});
    add("proj_out.weight", {d.channels, n});
    std::size_t at = 0;
    for (auto& s : out) {
        for (auto n : s.shape)
            s.count *= n;
        s.offset = at;
        at += s.count;
    }
    return out;
}

std::size_t locate(
    const std::vector<std::unique_ptr<checkpoint::Shard>>& shards,
    const std::string& name
) {
    std::size_t found = shards.size();
    for (std::size_t i = 0; i < shards.size(); ++i) {
        try {
            shards[i]->tensor(name);
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()) == "missing_tensor")
                continue;
            throw;
        }
        need(found == shards.size(), "image_duplicate_tensor");
        found = i;
    }
    need(found < shards.size(), "image_missing_tensor");
    return found;
}
}  // namespace

struct Denoiser::Impl {
    Resources& r;
    std::shared_ptr<DenseCompute> compute;
    std::unique_ptr<Lease> parser;
    std::vector<std::unique_ptr<checkpoint::Shard>> shards;
    std::string prefix;
    Lease metadata;
    DenoiserConfig d;
    std::vector<Spec> specs;
    std::map<std::string, std::size_t, std::less<>> names;
    std::unique_ptr<Buffer> weights;
    std::vector<std::unique_ptr<TransformerBlock>> blocks;

    Impl(Resources& r, DenoiserConfig config, std::shared_ptr<DenseCompute> c)
        : r(r), compute(std::move(c)), metadata(r, 16 * 1024 * 1024), d(config),
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
        const std::atomic_bool& cancel
    ) {
        const auto& spec = specs.at(names.at(name + ".weight"));
        need(
            spec.count == in * out && x.size() == rows * in && y.size() == rows * out,
            "image_projection_shape"
        );
        checkpoint::project_source(
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
            cancel
        );
        finite(y);
    }
};

Denoiser::Denoiser(std::shared_ptr<Resources> r, std::shared_ptr<DenseCompute> compute)
    : resources_(std::move(r)), compute_(std::move(compute)) {
    need(bool(resources_), "image_resources");
}

Denoiser::~Denoiser() {
    unload();
}

void Denoiser::load(
    const char* root,
    std::span<const std::string> files,
    DenoiserConfig d,
    const std::atomic_bool& cancel,
    const Hook& hook
) {
    Busy active(busy_);
    stop(cancel);
    need(!model_, "image_load_state");
    need(
        d.layers > 0 && d.layers <= 32 && d.context > 0 && d.context <= 4096
            && d.channels > 0 && d.channels <= 64 && d.block.state > 0
            && d.block.state <= 4096 && files.size() > 0 && files.size() <= 4,
        "image_denoiser_config"
    );
    need(
        std::set<std::string>(files.begin(), files.end()).size() == files.size(),
        "image_duplicate_shard"
    );
    auto m = std::make_unique<Impl>(*resources_, d, compute_);
    m->parser = std::make_unique<Lease>(*resources_, files.size() * 16 * 1024 * 1024);
    auto& shards = m->shards;
    for (const auto& file : files)
        shards.push_back(
            std::make_unique<checkpoint::Shard>(
                root,
                file,
                std::make_shared<checkpoint::MemoryBudget>(16 * 1024 * 1024),
                checkpoint::Limits{2 * 1024 * 1024, 4096, 4096}
            )
        );
    for (auto& s : m->specs) {
        s.shard = locate(shards, s.name);
        s.identity = shards[s.shard]->tensor_identity(s.name);
        auto t = shards[s.shard]->tensor(s.name);
        need(
            (t.dtype == checkpoint::Dtype::bf16 || t.dtype == checkpoint::Dtype::fp32)
                && t.rank == s.shape.size()
                && std::equal(s.shape.begin(), s.shape.end(), t.shape.begin()),
            "image_tensor_layout"
        );
    }
    m->weights = std::make_unique<Buffer>(*resources_, 1);
    for (std::size_t i = 0; i < d.layers; ++i) {
        stop(cancel);
        auto block = std::make_unique<TransformerBlock>(resources_, compute_);
        block->load(root, files, i, d.block, cancel);
        m->blocks.push_back(std::move(block));
        if (hook)
            hook("denoiser_block_loaded", i + 1);
    }
    for (const auto& shard : shards) {
        shard->check_unchanged();
    }
    stop(cancel);
    resources_->loaded(m->weights->lease.h);
    model_ = std::move(m);
}

void Denoiser::unload() {
    need(!busy_, "busy");
    if (model_) {
        resources_->begin_eviction(model_->weights->lease.h);
        model_.reset();
    }
}

std::optional<MediaPlanError> Denoiser::prepare_feed_forward(
    MediaSession& session,
    std::size_t block,
    std::size_t tokens,
    HiddenBoundary boundary,
    const std::atomic_bool& cancel
) {
    Busy active(busy_);
    stop(cancel);
    need(model_ && block < model_->blocks.size(), "image_load_state");
    auto& m = *model_;
    StageManifest manifest;
    manifest.prepare_projection = true;
    const bool final = block + 1 == m.blocks.size();
    const bool ingress = block == 0
                         && (boundary == HiddenBoundary::pre_attention
                             || boundary == HiddenBoundary::pre_qkv);
    manifest.prepare_hidden = ingress;
    for (const auto* name :
         {"img_in.weight",
          "txt_in.text_norm.weight",
          "txt_in.in_layer.weight",
          "txt_in.out_layer.weight",
          "time_text_embed.timestep_embedder.linear_1.weight",
          "time_text_embed.timestep_embedder.linear_2.weight",
          "modulation.1.weight",
          "norm_out.linear.weight",
          "proj_out.weight"}) {
        if (!ingress
            && (std::string_view(name).starts_with("txt_in.")
                || std::string_view(name) == "img_in.weight"))
            continue;
        if (!final
            && (std::string_view(name) == "norm_out.linear.weight"
                || std::string_view(name) == "proj_out.weight"))
            continue;
        const auto& spec = m.specs.at(m.names.at(name));
        auto& shard = *m.shards[spec.shard];
        shard.check_unchanged();
        const auto tensor = shard.tensor(spec.name);
        // Preserve project_source's STANDARD native-BF16 branch; F32 needs its
        // own resident implementation rather than adopting the FF policy.
        const bool norm = spec.name == "txt_in.text_norm.weight";
        if (!norm
            && (!compute_ || !compute_->supports_bf16_weights()
                || tensor.dtype != checkpoint::Dtype::bf16))
            return MediaPlanError{
                PlanFailure::unsupported_layout,
                (spec.name.starts_with("txt_in.") || spec.name == "img_in.weight")
                    ? "image_ingress_standard_precision"
                : (spec.name == "norm_out.linear.weight"
                   || spec.name == "proj_out.weight")
                    ? "image_final_standard_precision"
                : spec.name == "modulation.1.weight"
                    ? "image_modulation_standard_precision"
                    : "image_timestep_standard_precision",
                {}
            };
        need(
            shard.tensor_identity(spec.name) == spec.identity, "image_weight_identity"
        );
        manifest.weights.push_back(
            {spec.name,
             spec.identity,
             tensor.dtype == checkpoint::Dtype::bf16 ? TensorStorage::bf16_exact
                                                     : TensorStorage::f32,
             std::size_t(tensor.shape[0]),
             norm ? 1 : std::size_t(tensor.shape[1]),
             WeightPartition::replicated,
             1,
             1,
             norm ? ProjectionPrecision::f32 : ProjectionPrecision::bf16_f32}
        );
    }
    if (ingress) {
        manifest.tensors.insert(
            manifest.tensors.end(),
            {{"ingress/latent",
              tokens * m.d.channels * sizeof(float),
              MediaPhase::prepare,
              MediaPhase::prepare},
             {"ingress/condition",
              tokens * m.d.context * sizeof(float),
              MediaPhase::prepare,
              MediaPhase::prepare},
             {"ingress/normalized",
              tokens * m.d.context * sizeof(float),
              MediaPhase::prepare,
              MediaPhase::prepare},
             {"ingress/projected",
              tokens * m.d.block.state * sizeof(float),
              MediaPhase::prepare,
              MediaPhase::prepare}}
        );
    }
    manifest.tensors.push_back(
        {"conditioning/frequency",
         512 * sizeof(float),
         MediaPhase::prepare,
         MediaPhase::prepare}
    );
    manifest.tensors.push_back(
        {"conditioning/time-linear1",
         2 * m.d.block.state * sizeof(float),
         MediaPhase::prepare,
         MediaPhase::prepare}
    );
    manifest.tensors.push_back(
        {"conditioning/time-activated", 2 * m.d.block.state * sizeof(float)}
    );
    if (final) {
        manifest.tensors.insert(
            manifest.tensors.end(),
            {{"final/scale",
              m.d.block.state * sizeof(float),
              MediaPhase::complete,
              MediaPhase::complete},
             {"final/normalized",
              tokens * m.d.block.state * sizeof(float),
              MediaPhase::complete,
              MediaPhase::complete},
             {"final/projection-conversion",
              tokens * m.d.block.state * sizeof(std::uint16_t),
              MediaPhase::complete,
              MediaPhase::complete,
              AllocationKind::conversion},
             {"final/velocity",
              tokens * m.d.channels * sizeof(float),
              MediaPhase::complete,
              MediaPhase::complete}}
        );
    }
    if (auto error = m.blocks[block]->prepare_feed_forward(
            session, tokens, boundary, cancel, std::move(manifest)
        ))
        return error;
    try {
        for (const auto& slice : session.plan().weights) {
            if (!m.names.contains(slice.name))
                continue;
            const auto& spec = m.specs.at(m.names.at(slice.name));
            auto& shard = *m.shards[spec.shard];
            stop(cancel);
            shard.check_unchanged();
            auto destination = session.weight(slice);
            if (session.ready(destination))
                continue;
            Lease staging(*resources_, slice.stored_bytes);
            std::vector<std::uint8_t> raw(slice.stored_bytes);
            shard.read_tensor(spec.name, 0, raw);
            shard.check_unchanged();
            stop(cancel);
            // The standard host source validates raw weights, even when finite
            // output cancellation could otherwise hide a nonfinite weight.
            const auto dtype = shard.tensor(spec.name).dtype;
            for (std::size_t i = 0; i < raw.size();
                 i += dtype == checkpoint::Dtype::bf16 ? 2 : 4)
                need(
                    dtype == checkpoint::Dtype::fp32
                        ? ((std::uint32_t(raw[i + 3]) << 24
                            | std::uint32_t(raw[i + 2]) << 16
                            | std::uint32_t(raw[i + 1]) << 8 | raw[i])
                           & 0x7f800000)
                              != 0x7f800000
                        : ((std::uint16_t(raw[i + 1]) << 8 | raw[i]) & 0x7f80)
                              != 0x7f80,
                    "h3_cuda_nonfinite_weight"
                );
            session.upload_weight(destination, std::as_bytes(std::span(raw)));
        }
    } catch (...) {
        session.abort(std::current_exception());
        throw;
    }
    return {};
}

void Denoiser::execute_ingress(
    MediaSession& session,
    std::span<const float> latent,
    std::span<const float> condition,
    std::size_t text,
    std::span<video::ImageFeedForwardLauncher* const> launchers,
    const std::atomic_bool& cancel
) {
    Busy active(busy_);
    stop(cancel);
    need(bool(model_), "image_load_state");
    const auto state = model_->d.block.state, context = model_->d.context,
               channels = model_->d.channels;

    struct Device {
        video::ImageFeedForwardLauncher* launcher;
        DeviceTensorView latent, condition, norm_weight, normalized, projected,
            text_weight, output_weight, image_weight, hidden, workspace, status;
    };

    try {
        need(
            text && text <= 2048 && !latent.empty() && latent.size() % channels == 0,
            "image_ingress_shape"
        );
        const auto images = latent.size() / channels, tokens = text + images;
        need(
            images <= 65536 && tokens <= 32768 && condition.size() == text * context
                && session.plan().schedule == MediaSchedule::feed_forward
                && launchers.size() == session.plan().devices.size(),
            "image_ingress_shape"
        );
        finite(latent);
        finite(condition);

        struct Projection {
            DeviceTensorView *input, *weight, *output;
            std::size_t first, rows;
            bool gelu;
        };

        auto projections = [&](Device& d) {
            return std::array{
                Projection{&d.normalized, &d.text_weight, &d.projected, 0, text, true},
                Projection{&d.projected, &d.output_weight, &d.hidden, 0, text, false},
                Projection{&d.latent, &d.image_weight, &d.hidden, text, images, false}
            };
        };
        std::vector<Device> devices;
        for (const auto& planned : session.plan().devices) {
            video::ImageFeedForwardLauncher* launcher = nullptr;
            for (auto* candidate : launchers) {
                need(candidate, "image_ingress_launcher");
                if (candidate->device() == planned.device) {
                    need(!launcher, "image_ingress_launcher");
                    launcher = candidate;
                }
            }
            need(launcher, "image_ingress_launcher");
            const auto hidden_boundary = std::find_if(
                session.plan().tensors.begin(),
                session.plan().tensors.end(),
                [&](const auto& t) {
                    return t.device == planned.device && t.name == "resident/hidden";
                }
            );
            need(
                hidden_boundary != session.plan().tensors.end()
                    && hidden_boundary->bytes == tokens * state * sizeof(float),
                "image_ingress_shape"
            );
            auto matrix = [&](const char* name,
                              std::size_t rows,
                              std::size_t columns,
                              DeviceDType dtype = DeviceDType::f32) {
                return session.tensor(
                    planned.device, name, dtype, {rows, columns}, {columns, 1}
                );
            };
            auto weight = [&](const char* name, std::size_t rows, std::size_t columns) {
                const auto& spec = model_->specs.at(model_->names.at(name));
                const auto width = model_->shards[spec.shard]->tensor(spec.name).dtype
                                           == checkpoint::Dtype::fp32
                                       ? 4u
                                       : 2u;
                const auto found = std::find_if(
                    session.plan().weights.begin(),
                    session.plan().weights.end(),
                    [&](const auto& w) {
                        return w.device == planned.device && w.name == spec.name
                               && w.identity == spec.identity && w.first_row == 0
                               && w.rows == rows
                               && w.stored_bytes == rows * columns * width;
                    }
                );
                need(found != session.plan().weights.end(), "image_ingress_weight");
                return session.weight(*found);
            };
            devices.push_back(
                {launcher,
                 matrix("ingress/latent", images, channels),
                 matrix("ingress/condition", text, context),
                 weight("txt_in.text_norm.weight", context, 1),
                 matrix("ingress/normalized", text, context),
                 matrix("ingress/projected", text, state),
                 weight("txt_in.in_layer.weight", state, context),
                 weight("txt_in.out_layer.weight", state, state),
                 weight("img_in.weight", state, channels),
                 matrix("resident/hidden", tokens, state),
                 session.tensor(
                     planned.device,
                     "resident/blas",
                     DeviceDType::u8,
                     {video::bf16_projection_workspace_bytes},
                     {1}
                 ),
                 session.tensor(
                     planned.device, "resident/status", DeviceDType::u32, {1}, {1}
                 )}
            );
            auto& d = devices.back();
            const std::array ready{
                &d.norm_weight, &d.text_weight, &d.output_weight, &d.image_weight
            };
            session.require_ready(ready);
            video::validate_image_text_normalization(
                planned.device,
                d.condition.layout(),
                d.norm_weight.layout(),
                d.normalized.layout(),
                d.status.layout()
            );
            const MediaDeviceSpan bias{
                nullptr, planned.device, DeviceDType::f32, {}, {}, 0, 0, 0
            };
            for (const auto& projection : projections(d)) {
                auto input = session.rows(*projection.input, 0, 1);
                auto output = session.rows(*projection.output, projection.first, 1);
                auto rounded = matrix(
                    "resident/projection-conversion",
                    1,
                    input.layout().dimensions[1],
                    DeviceDType::bf16
                );
                video::validate_bf16_projection(
                    planned.device,
                    input.layout(),
                    projection.weight->layout(),
                    bias,
                    rounded.layout(),
                    output.layout(),
                    d.workspace.layout(),
                    d.status.layout()
                );
            }
        }
        // Validate every device before uploading input or submitting any producer.
        for (auto& d : devices) {
            stop(cancel);
            session.upload_input(d.latent, std::as_bytes(latent));
            session.upload_input(d.condition, std::as_bytes(condition));
            stop(cancel);
            const std::array read{&d.condition, &d.norm_weight};
            const std::array write{&d.normalized, &d.status};
            session.launch(
                d.launcher->device(),
                read,
                write,
                [&](void* stream, auto in, auto out) {
                    d.launcher->text_normalization(stream, in, out);
                }
            );
            session.complete(write, d.status);
            // Serial rows reuse the smallest admitted conversion tile. Text/image
            // hidden regions certify contiguous coverage under the same owner.
            for (const auto& projection : projections(d)) {
                for (std::size_t row = 0; row < projection.rows; ++row) {
                    stop(cancel);
                    auto input = session.rows(*projection.input, row, 1);
                    auto output =
                        session.rows(*projection.output, projection.first + row, 1);
                    auto rounded = session.tensor(
                        d.launcher->device(),
                        "resident/projection-conversion",
                        DeviceDType::bf16,
                        {1, input.layout().dimensions[1]},
                        {input.layout().dimensions[1], 1}
                    );
                    const std::array projection_read{&input, projection.weight};
                    const std::array projection_write{
                        &output, &rounded, &d.workspace, &d.status
                    };
                    session.launch(
                        d.launcher->device(),
                        projection_read,
                        projection_write,
                        [&](void* stream, auto in, auto out) {
                            if (projection.gelu)
                                d.launcher->text_projection(stream, in, out);
                            else
                                d.launcher->standard(stream, in, out);
                        }
                    );
                    session.complete(projection_write, d.status);
                }
            }
            stop(cancel);
            const std::array hidden{&d.hidden};
            session.require_ready(hidden);
        }
    } catch (...) {
        session.abort(std::current_exception());
        throw;
    }
}

void Denoiser::execute_feed_forward(
    MediaSession& session,
    std::size_t block,
    std::size_t tokens,
    std::size_t text,
    std::optional<float> timestep,
    HiddenBoundary boundary,
    std::span<video::ImageFeedForwardLauncher* const> launchers,
    const std::atomic_bool& cancel
) {
    Busy active(busy_);
    need(
        boundary == HiddenBoundary::pre_attention
            || boundary == HiddenBoundary::post_attention,
        "image_ff_plan"
    );
    execute_conditioning(
        session, block, tokens, text, timestep, boundary, launchers, cancel
    );
    const bool attention = boundary == HiddenBoundary::pre_attention;
    const bool final = block + 1 == model_->blocks.size();
    const auto& plan = session.plan();
    try {
        auto& transformer = *model_->blocks[block];
        transformer.prepare_feed_forward_inputs(
            session, tokens, text, boundary, launchers, cancel
        );
        transformer.execute_feed_forward(
            session,
            tokens,
            text,
            launchers,
            cancel,
            ((final ? 5 : 3) + (block == 0 && attention ? 4 : 0)) * plan.devices.size()
        );
        if (final)
            execute_final_output(session, tokens, text, launchers, cancel);
    } catch (...) {
        session.abort(std::current_exception());
        throw;
    }
}

void Denoiser::execute_qkv(
    MediaSession& session,
    std::size_t block,
    std::size_t tokens,
    std::size_t text,
    std::optional<float> timestep,
    std::span<video::ImageFeedForwardLauncher* const> launchers,
    const std::atomic_bool& cancel
) {
    Busy active(busy_);
    execute_conditioning(
        session,
        block,
        tokens,
        text,
        timestep,
        HiddenBoundary::pre_qkv,
        launchers,
        cancel
    );
    model_->blocks[block]->execute_qkv(session, tokens, text, launchers, cancel);
}

void Denoiser::execute_conditioning(
    MediaSession& session,
    std::size_t block,
    std::size_t tokens,
    std::size_t text,
    std::optional<float> timestep,
    HiddenBoundary boundary,
    std::span<video::ImageFeedForwardLauncher* const> launchers,
    const std::atomic_bool& cancel
) {
    stop(cancel);
    need(model_ && block < model_->blocks.size(), "image_load_state");
    const auto state = model_->d.block.state;
    const auto& plan = session.plan();
    const bool qkv = boundary == HiddenBoundary::pre_qkv;
    const bool attention = qkv || boundary == HiddenBoundary::pre_attention;
    const bool final = block + 1 == model_->blocks.size();

    struct Device {
        video::ImageFeedForwardLauncher* launcher;
        DeviceTensorView frequency, linear1, time, modulation, linear1_weight,
            time_weight, modulation_weight;
    };

    try {
        need(
            tokens && text <= tokens && (!final || text < tokens) && plan.rows
                && (attention || boundary == HiddenBoundary::post_attention)
                && plan.schedule == MediaSchedule::feed_forward
                && launchers.size() == plan.devices.size()
                && plan.weights.size()
                       == ((qkv         ? 10
                            : attention ? 7
                                        : 6)
                           + (final ? 2 : 0) + (block == 0 && attention ? 4 : 0))
                              * plan.devices.size(),
            "image_ff_plan"
        );
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
            auto weight = [&](const char* name, std::size_t rows, std::size_t columns) {
                const auto& spec = model_->specs.at(model_->names.at(name));
                const auto slice = std::find_if(
                    plan.weights.begin(), plan.weights.end(), [&](const auto& w) {
                        return w.device == planned.device && w.name == spec.name
                               && w.identity == spec.identity && w.first_row == 0
                               && w.rows == rows
                               && w.stored_bytes == rows * columns * 2;
                    }
                );
                need(slice != plan.weights.end(), "image_conditioning_weight");
                return session.weight(*slice);
            };
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
                 matrix("conditioning/frequency", 2, 256),
                 matrix("conditioning/time-linear1", 2, state),
                 matrix("conditioning/time-activated", 2, state),
                 matrix("ff/modulation", 2, 4 * state),
                 weight(
                     "time_text_embed.timestep_embedder.linear_1.weight", state, 256
                 ),
                 weight(
                     "time_text_embed.timestep_embedder.linear_2.weight", state, state
                 ),
                 weight("modulation.1.weight", 4 * state, state)}
            );
            auto& d = devices.back();
            auto hidden = matrix("resident/hidden", tokens, state);
            auto normalized = matrix("resident/normalized", tokens, state);
            auto gates = matrix("ff/residual-gates", 2, state);
            auto status = session.tensor(
                planned.device, "resident/status", DeviceDType::u32, {1}, {1}
            );
            const std::array ready{
                &d.linear1_weight, &d.time_weight, &d.modulation_weight, &hidden
            };
            session.require_ready(ready);
            if (final) {
                auto final_weight = weight("norm_out.linear.weight", state, state);
                auto output_weight =
                    weight("proj_out.weight", model_->d.channels, state);
                const std::array final_ready{&final_weight, &output_weight};
                session.require_ready(final_ready);
            }
            if (timestep)
                video::validate_image_frequency(
                    planned.device, d.frequency.layout(), status.layout(), *timestep
                );
            else {
                const std::array conditioning{&d.time, &d.modulation};
                session.require_ready(conditioning);
            }
            const std::array inputs{&d.frequency, &d.linear1, &d.time};
            const std::array outputs{&d.linear1, &d.time, &d.modulation};
            const std::array weights{
                &d.linear1_weight, &d.time_weight, &d.modulation_weight
            };
            auto workspace = session.tensor(
                planned.device,
                "resident/blas",
                DeviceDType::u8,
                {video::bf16_projection_workspace_bytes},
                {1}
            );
            const MediaDeviceSpan bias{
                nullptr, planned.device, DeviceDType::f32, {}, {}, 0, 0, 0
            };
            for (std::size_t stage = 0; stage < weights.size(); ++stage) {
                const auto columns = stage == 0 ? 256 : state;
                auto input = session.rows(*inputs[stage], 0, 1);
                auto output = session.rows(*outputs[stage], 0, 1);
                auto rounded = session.tensor(
                    planned.device,
                    "resident/projection-conversion",
                    DeviceDType::bf16,
                    {1, columns},
                    {columns, 1}
                );
                video::validate_bf16_projection(
                    planned.device,
                    input.layout(),
                    weights[stage]->layout(),
                    bias,
                    rounded.layout(),
                    output.layout(),
                    workspace.layout(),
                    status.layout()
                );
            }
            if (attention && !qkv) {
                const std::array attended{&normalized};
                session.require_ready(attended);
                video::validate_image_attention_residual(
                    planned.device,
                    hidden.layout(),
                    normalized.layout(),
                    d.modulation.layout(),
                    hidden.layout(),
                    gates.layout(),
                    status.layout(),
                    text
                );
            }
            video::validate_image_modulation(
                planned.device,
                hidden.layout(),
                d.modulation.layout(),
                normalized.layout(),
                gates.layout(),
                status.layout(),
                text
            );
        }
        // All ingress and consumer layouts are checked before any producer runs.
        for (auto& d : devices) {
            if (!timestep)
                continue;
            stop(cancel);
            {
                auto status = session.tensor(
                    d.launcher->device(), "resident/status", DeviceDType::u32, {1}, {1}
                );
                const std::array<DeviceTensorView*, 0> read{};
                const std::array write{&d.frequency, &status};
                session.launch(
                    d.launcher->device(),
                    read,
                    write,
                    [&](void* stream, auto, auto out) {
                        d.launcher->frequency(stream, out, *timestep);
                    }
                );
                session.complete(write, status);
            }
            // Complete each producer before its consumer. One row at a time fits
            // the smallest admitted projection tile, reusing conversion/BLAS scratch.
            const std::array inputs{&d.frequency, &d.linear1, &d.time};
            const std::array outputs{&d.linear1, &d.time, &d.modulation};
            const std::array weights{
                &d.linear1_weight, &d.time_weight, &d.modulation_weight
            };
            for (std::size_t stage = 0; stage < weights.size(); ++stage) {
                for (std::size_t row = 0; row < 2; ++row) {
                    stop(cancel);
                    const auto device = d.launcher->device();
                    auto input = session.rows(*inputs[stage], row, 1);
                    auto output = session.rows(*outputs[stage], row, 1);
                    const auto columns = stage == 0 ? 256 : state;
                    auto rounded = session.tensor(
                        device,
                        "resident/projection-conversion",
                        DeviceDType::bf16,
                        {1, columns},
                        {columns, 1}
                    );
                    auto workspace = session.tensor(
                        device,
                        "resident/blas",
                        DeviceDType::u8,
                        {video::bf16_projection_workspace_bytes},
                        {1}
                    );
                    auto status = session.tensor(
                        device, "resident/status", DeviceDType::u32, {1}, {1}
                    );
                    const std::array read{&input, weights[stage]};
                    const std::array write{&output, &rounded, &workspace, &status};
                    session.launch(
                        device, read, write, [&](void* stream, auto in, auto out) {
                            if (stage < 2)
                                d.launcher->timestep(stream, in, out);
                            else
                                d.launcher->standard(stream, in, out);
                        }
                    );
                    session.complete(write, status);
                }
            }
        }
        devices.clear();
    } catch (...) {
        session.abort(std::current_exception());
        throw;
    }
}

void Denoiser::execute_final_output(
    MediaSession& session,
    std::size_t tokens,
    std::size_t text,
    std::span<video::ImageFeedForwardLauncher* const> launchers,
    const std::atomic_bool& cancel
) {
    const auto state = model_->d.block.state, channels = model_->d.channels;
    const auto& spec = model_->specs.at(model_->names.at("norm_out.linear.weight"));
    const auto& output_spec = model_->specs.at(model_->names.at("proj_out.weight"));

    struct Device {
        video::ImageFeedForwardLauncher* launcher;
        DeviceTensorView time, weight, output_weight, hidden, scale, normalized,
            rounded, velocity, workspace, status;
    };

    std::vector<Device> devices;
    for (const auto& planned : session.plan().devices) {
        auto* launcher =
            *std::find_if(launchers.begin(), launchers.end(), [&](auto* candidate) {
                return candidate->device() == planned.device;
            });
        const auto slice = std::find_if(
            session.plan().weights.begin(),
            session.plan().weights.end(),
            [&](const auto& w) {
                return w.device == planned.device && w.name == spec.name
                       && w.identity == spec.identity && w.first_row == 0
                       && w.rows == state && w.stored_bytes == state * state * 2;
            }
        );
        need(slice != session.plan().weights.end(), "image_final_weight");
        const auto output_slice = std::find_if(
            session.plan().weights.begin(),
            session.plan().weights.end(),
            [&](const auto& w) {
                return w.device == planned.device && w.name == output_spec.name
                       && w.identity == output_spec.identity && w.first_row == 0
                       && w.rows == channels && w.stored_bytes == channels * state * 2;
            }
        );
        need(output_slice != session.plan().weights.end(), "image_final_weight");
        auto matrix = [&](const char* name,
                          std::size_t rows,
                          DeviceDType dtype = DeviceDType::f32) {
            return session.tensor(
                planned.device, name, dtype, {rows, state}, {state, 1}
            );
        };
        auto time = matrix("conditioning/time-activated", 2);
        auto hidden = matrix("resident/hidden", tokens);
        devices.push_back(
            {launcher,
             session.rows(time, 0, 1),
             session.weight(*slice),
             session.weight(*output_slice),
             session.rows(hidden, text, tokens - text),
             matrix("final/scale", 1),
             matrix("final/normalized", tokens - text),
             matrix("final/projection-conversion", tokens - text, DeviceDType::bf16),
             session.tensor(
                 planned.device,
                 "final/velocity",
                 DeviceDType::f32,
                 {tokens - text, channels},
                 {channels, 1}
             ),
             session.tensor(
                 planned.device,
                 "resident/blas",
                 DeviceDType::u8,
                 {video::bf16_projection_workspace_bytes},
                 {1}
             ),
             session.tensor(
                 planned.device, "resident/status", DeviceDType::u32, {1}, {1}
             )}
        );
        auto& d = devices.back();
        const std::array ready{&d.time, &d.weight, &d.output_weight, &d.hidden};
        auto rounded = session.rows(d.rounded, 0, 1);
        session.require_ready(ready);
        const MediaDeviceSpan bias{
            nullptr, planned.device, DeviceDType::f32, {}, {}, 0, 0, 0
        };
        video::validate_bf16_projection(
            planned.device,
            d.time.layout(),
            d.weight.layout(),
            bias,
            rounded.layout(),
            d.scale.layout(),
            d.workspace.layout(),
            d.status.layout()
        );
        video::validate_image_final_normalization(
            planned.device,
            d.hidden.layout(),
            d.scale.layout(),
            d.normalized.layout(),
            d.status.layout()
        );
        video::validate_bf16_projection(
            planned.device,
            d.normalized.layout(),
            d.output_weight.layout(),
            bias,
            d.rounded.layout(),
            d.velocity.layout(),
            d.workspace.layout(),
            d.status.layout()
        );
    }
    // All final ingress/layouts precede the first norm_out submission.
    for (auto& d : devices) {
        stop(cancel);
        const std::array read{&d.time, &d.weight};
        auto rounded = session.rows(d.rounded, 0, 1);
        const std::array write{&d.scale, &rounded, &d.workspace, &d.status};
        session.launch(
            d.launcher->device(), read, write, [&](void* stream, auto in, auto out) {
                d.launcher->standard(stream, in, out);
            }
        );
        session.complete(write, d.status);
        stop(cancel);
        const std::array input{&d.hidden, &d.scale};
        const std::array output{&d.normalized, &d.status};
        session.launch(
            d.launcher->device(), input, output, [&](void* stream, auto in, auto out) {
                d.launcher->final_normalization(stream, in, out);
            }
        );
        session.complete(output, d.status);
        stop(cancel);
        const std::array projection_input{&d.normalized, &d.output_weight};
        const std::array projection_output{
            &d.velocity, &d.rounded, &d.workspace, &d.status
        };
        session.launch(
            d.launcher->device(),
            projection_input,
            projection_output,
            [&](void* stream, auto in, auto out) {
                d.launcher->standard(stream, in, out);
            }
        );
        session.complete(projection_output, d.status);
        stop(cancel);
    }
}

void Denoiser::execute(
    std::span<const float> latent,
    std::span<const float> condition,
    std::size_t text,
    std::size_t height,
    std::size_t width,
    float timestep,
    std::span<float> velocity,
    const std::atomic_bool& cancel,
    const Hook& hook
) {
    Busy active(busy_);
    stop(cancel);
    need(bool(model_), "image_not_loaded");
    auto& m = *model_;
    auto d = m.d;
    const auto state = d.block.state;
    need(
        text > 0 && text <= 2048 && height > 0 && height <= 256 && width > 0
            && width <= 256 && height * width % 4 == 0 && text + height * width <= 32768
            && std::isfinite(timestep) && timestep >= 0 && timestep <= 1,
        "image_denoiser_layout"
    );
    const auto images = height * width, rows = text + images;
    need(
        latent.size() == images * d.channels && velocity.size() == latent.size()
            && condition.size() == text * d.context,
        "image_shape"
    );
    finite(latent);
    finite(condition);
    Pin pin(*resources_, m.weights->lease.h);
    Buffer normalized(*resources_, text * d.context),
        projected(*resources_, text * state), x(*resources_, rows * state),
        next(*resources_, rows * state), frequency(*resources_, 512),
        time1(*resources_, 2 * state), time2(*resources_, 2 * state),
        modulation(*resources_, 8 * state), scale(*resources_, state),
        final(*resources_, images * state), out(*resources_, velocity.size());
    auto norm = m.weight("txt_in.text_norm.weight", cancel);
    for (std::size_t t = 0; t < text; ++t) {
        stop(cancel);
        double mean = 0;
        for (std::size_t c = 0; c < d.context; ++c)
            mean += double(condition[t * d.context + c]) * condition[t * d.context + c];
        float s = float(1 / std::sqrt(mean / d.context + 1e-6));
        for (std::size_t c = 0; c < d.context; ++c)
            normalized.data[t * d.context + c] =
                condition[t * d.context + c] * s * (1 + norm[c]);
    }
    m.linear(
        "txt_in.in_layer",
        normalized.span(),
        projected.span(),
        text,
        d.context,
        state,
        cancel
    );
    for (float& v : projected.span())
        v = .5f * v * (1 + std::tanh(.7978845608028654f * (v + .044715f * v * v * v)));
    m.linear(
        "txt_in.out_layer",
        projected.span(),
        x.span().first(text * state),
        text,
        state,
        state,
        cancel
    );
    m.linear(
        "img_in",
        latent,
        x.span().subspan(text * state),
        images,
        d.channels,
        state,
        cancel
    );
    for (std::size_t row = 0; row < 2; ++row)
        for (std::size_t c = 0; c < 256; ++c)
            frequency.data[row * 256 + c] =
                detail::timestep_frequency(row ? 0.f : timestep, c);
    m.linear(
        "time_text_embed.timestep_embedder.linear_1",
        frequency.span(),
        time1.span(),
        2,
        256,
        state,
        cancel
    );
    for (float& v : time1.span())
        v = detail::silu(v);
    m.linear(
        "time_text_embed.timestep_embedder.linear_2",
        time1.span(),
        time2.span(),
        2,
        state,
        state,
        cancel
    );
    for (float& v : time2.span())
        v = detail::silu(v);
    m.linear(
        "modulation.1", time2.span(), modulation.span(), 2, state, 4 * state, cancel
    );
    if (hook)
        hook("conditioning", 0);
    for (std::size_t i = 0; i < d.layers; ++i) {
        if (hook)
            hook("block_started", i);
        m.blocks[i]->execute(
            x.span(),
            modulation.span(),
            text,
            height,
            width,
            next.span(),
            cancel,
            [&](const char* phase) {
                if (hook)
                    hook(phase, i);
            }
        );
        std::swap(x.data, next.data);
        if (hook)
            hook("block", i);
        stop(cancel);
    }
    m.linear(
        "norm_out.linear",
        time2.span().first(state),
        scale.span(),
        1,
        state,
        state,
        cancel
    );
    for (std::size_t t = 0; t < images; ++t) {
        stop(cancel);
        auto input = x.span().subspan((text + t) * state, state);
        double mean = 0, variance = 0;
        for (float v : input)
            mean += v;
        mean /= state;
        for (float v : input)
            variance += (v - mean) * (v - mean);
        float n = float(1 / std::sqrt(variance / state + 1e-6));
        for (std::size_t c = 0; c < state; ++c)
            final.data[t * state + c] =
                float(input[c] - mean) * n * (1 + scale.data[c]);
    }
    m.linear("proj_out", final.span(), out.span(), images, state, d.channels, cancel);
    stop(cancel);
    std::copy(out.span().begin(), out.span().end(), velocity.begin());
}
}  // namespace kadan::image
