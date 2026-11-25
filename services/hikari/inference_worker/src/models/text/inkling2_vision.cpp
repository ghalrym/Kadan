#include "inkling2_vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <png.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace kadan::text {
namespace {
constexpr std::size_t v_image_size = 640;
constexpr std::size_t v_patch_size = 20;
constexpr std::size_t v_merge = 2;
constexpr std::size_t v_align = v_patch_size * v_merge;  // 40
constexpr std::size_t v_grid_base = v_image_size / v_patch_size;  // 32
constexpr std::size_t v_max_merged_tokens = 4096;
constexpr std::size_t v_max_pixels = v_max_merged_tokens * v_align * v_align;
constexpr std::size_t v_layers = 23;
constexpr std::size_t v_embd = 1024;
constexpr std::size_t v_ff = 4096;
constexpr std::size_t v_heads = 16;
constexpr std::size_t v_head_dim = v_embd / v_heads;  // 64
constexpr std::size_t v_proj_dim = 2304;
constexpr float v_ln_eps = 1e-5f;
constexpr float v_rope_theta = 10000.0f;

std::string vblk(std::size_t i, const char* suffix) {
    return "v.blk." + std::to_string(i) + "." + suffix;
}

bool close_to(double actual, double expected, double tol) {
    return std::isfinite(actual) && std::abs(actual - expected) <= tol;
}

struct PngReader {
    png_image image{};

    explicit PngReader(std::span<const std::uint8_t> bytes) {
        image.version = PNG_IMAGE_VERSION;
        if (bytes.empty()
            || !png_image_begin_read_from_memory(&image, bytes.data(), bytes.size())) {
            png_image_free(&image);
            throw std::runtime_error("inkling2v_invalid_png");
        }
    }

    ~PngReader() {
        png_image_free(&image);
    }
};

struct RgbImage {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<std::uint8_t> rgb;  // [height * width * 3]
};

RgbImage decode_png_rgb(std::span<const std::uint8_t> png_bytes) {
    PngReader reader(png_bytes);
    auto& image = reader.image;
    need(
        image.width >= 1 && image.height >= 1 && image.width <= 16384
            && image.height <= 16384,
        "inkling2v_png_dimensions"
    );
    image.format = PNG_FORMAT_RGB;
    RgbImage out;
    out.width = image.width;
    out.height = image.height;
    out.rgb.resize(out.width * out.height * 3);
    need(
        png_image_finish_read(&image, nullptr, out.rgb.data(), 0, nullptr) != 0,
        "inkling2v_png_decode"
    );
    return out;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

RgbImage resize_bilinear_rgb(
    const RgbImage& src, std::size_t dst_w, std::size_t dst_h
) {
    RgbImage dst;
    dst.width = dst_w;
    dst.height = dst_h;
    dst.rgb.resize(dst_w * dst_h * 3);
    const float scale_x = float(src.width) / float(dst_w);
    const float scale_y = float(src.height) / float(dst_h);
    for (std::size_t y = 0; y < dst_h; ++y) {
        const float fy = std::clamp((float(y) + 0.5f) * scale_y - 0.5f, 0.0f, float(src.height - 1));
        const std::size_t y0 = std::size_t(std::floor(fy));
        const std::size_t y1 = std::min(y0 + 1, src.height - 1);
        const float wy = fy - float(y0);
        for (std::size_t x = 0; x < dst_w; ++x) {
            const float fx = std::clamp((float(x) + 0.5f) * scale_x - 0.5f, 0.0f, float(src.width - 1));
            const std::size_t x0 = std::size_t(std::floor(fx));
            const std::size_t x1 = std::min(x0 + 1, src.width - 1);
            const float wx = fx - float(x0);
            for (std::size_t c = 0; c < 3; ++c) {
                const float v00 = float(src.rgb[(y0 * src.width + x0) * 3 + c]);
                const float v01 = float(src.rgb[(y0 * src.width + x1) * 3 + c]);
                const float v10 = float(src.rgb[(y1 * src.width + x0) * 3 + c]);
                const float v11 = float(src.rgb[(y1 * src.width + x1) * 3 + c]);
                const float v = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01)
                                + wy * ((1.0f - wx) * v10 + wx * v11);
                dst.rgb[(y * dst_w + x) * 3 + c] =
                    std::uint8_t(std::clamp(std::lround(v), 0L, 255L));
            }
        }
    }
    return dst;
}

void layer_norm(
    std::span<float> x,
    std::span<const float> weight,
    std::span<const float> bias,
    float eps
) {
    const std::size_t width = weight.size();
    need(width && bias.size() == width && x.size() % width == 0, "inkling2v_ln_shape");
    for (std::size_t at = 0; at < x.size(); at += width) {
        double mean = 0.0;
        for (std::size_t i = 0; i < width; ++i)
            mean += double(x[at + i]);
        mean /= double(width);
        double var = 0.0;
        for (std::size_t i = 0; i < width; ++i) {
            const double d = double(x[at + i]) - mean;
            var += d * d;
        }
        const float inv = 1.0f / std::sqrt(float(var / double(width)) + eps);
        const float m = float(mean);
        for (std::size_t i = 0; i < width; ++i)
            x[at + i] = (x[at + i] - m) * inv * weight[i] + bias[i];
    }
}

void add_bias(std::span<float> x, std::span<const float> bias) {
    const std::size_t width = bias.size();
    for (std::size_t at = 0; at < x.size(); at += width)
        for (std::size_t i = 0; i < width; ++i)
            x[at + i] += bias[i];
}

float gelu_erf(float x) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + std::erff(x * inv_sqrt2));
}

// 2D axial RoPE matching mtmd_graph::build_rope_2d(..., freq_base=10000, interleave=false):
// First half [0..32) uses paired (2i, 2i+1) rotation with pos_w;
// Second half [32..64) uses paired (32+2i, 32+2i+1) rotation with pos_h.
void apply_rope_2d(
    std::span<float> x,
    std::size_t n_pos,
    const std::vector<int>& pos_w,
    const std::vector<int>& pos_h
) {
    constexpr std::size_t half_dim = v_head_dim / 2;  // 32
    constexpr std::size_t pairs = half_dim / 2;       // 16
    std::array<float, pairs> inv_freq;
    for (std::size_t i = 0; i < pairs; ++i)
        inv_freq[i] = std::pow(v_rope_theta, -2.0f * float(i) / float(half_dim));

    for (std::size_t p = 0; p < n_pos; ++p) {
        const float pw = float(pos_w[p]);
        const float ph = float(pos_h[p]);
        for (std::size_t h = 0; h < v_heads; ++h) {
            float* ptr = x.data() + (p * v_heads + h) * v_head_dim;
            for (std::size_t i = 0; i < pairs; ++i) {
                const float tw = pw * inv_freq[i];
                const float cw = std::cos(tw), sw = std::sin(tw);
                const float a0 = ptr[2 * i], a1 = ptr[2 * i + 1];
                ptr[2 * i] = a0 * cw - a1 * sw;
                ptr[2 * i + 1] = a0 * sw + a1 * cw;

                const float th = ph * inv_freq[i];
                const float ch = std::cos(th), sh = std::sin(th);
                const float b0 = ptr[half_dim + 2 * i], b1 = ptr[half_dim + 2 * i + 1];
                ptr[half_dim + 2 * i] = b0 * ch - b1 * sh;
                ptr[half_dim + 2 * i + 1] = b0 * sh + b1 * ch;
            }
        }
    }
}

struct VisionBlock {
    Vec ln1_w, ln1_b, ln2_w, ln2_b;
    Vec ls1_w, ls2_w;
    Vec q_b, k_b, v_b, out_b, up_b, down_b;
    std::string q_w, k_w, v_w, out_w, up_w, down_w;
};
}  // namespace

struct InklingVision::Impl {
    std::unique_ptr<GgufWeights> weights;
    Vec patch_embd;      // [v_embd, 3, 20, 20]
    Vec position_embd;   // [32 * 32, v_embd]
    Vec class_embd;      // [v_embd]
    Vec class_pos_embd;  // [v_embd]
    Vec pre_ln_w, pre_ln_b;
    std::vector<VisionBlock> blocks;
    std::vector<WeightIdentity> matrix_identities;

    Impl(const std::string& mmproj_path, const std::atomic_bool* cancel) {
        weights = std::make_unique<GgufWeights>(mmproj_path.c_str(), cancel);
        weights->set_ram_budget(0);
        const auto& g = weights->gguf();

        need(g.string("general.architecture") == "clip", "inkling2v_architecture");
        need(
            g.string("clip.vision.projector_type") == "inkling2v",
            "inkling2v_projector_type"
        );
        need(
            g.integer("clip.vision.image_size") == v_image_size
                && g.integer("clip.vision.patch_size") == v_patch_size
                && g.integer("clip.vision.spatial_merge_size") == v_merge
                && g.integer("clip.vision.block_count") == v_layers
                && g.integer("clip.vision.embedding_length") == v_embd
                && g.integer("clip.vision.feed_forward_length") == v_ff
                && g.integer("clip.vision.attention.head_count") == v_heads
                && g.integer("clip.vision.projection_dim") == v_proj_dim
                && close_to(
                    g.number("clip.vision.attention.layer_norm_epsilon"), v_ln_eps, 1e-7
                ),
            "inkling2v_metadata"
        );

        weights->vector("v.patch_embd.weight", {v_patch_size, v_patch_size, 3, v_embd});
        weights->table("v.position_embd.weight", v_grid_base * v_grid_base, v_embd);
        weights->vector("v.class_embd", {v_embd});
        weights->vector("v.class_position_embd.weight", {v_embd});
        weights->vector("v.pre_ln.weight", {v_embd});
        weights->vector("v.pre_ln.bias", {v_embd});

        auto track_matrix = [&](const std::string& name, std::size_t rows, std::size_t cols) {
            weights->matrix(name, rows, cols);
            matrix_identities.push_back(g.identity(name, 0));
        };

        blocks.resize(v_layers);
        for (std::size_t i = 0; i < v_layers; ++i) {
            auto& b = blocks[i];
            weights->vector(vblk(i, "ln1.weight"), {v_embd});
            weights->vector(vblk(i, "ln1.bias"), {v_embd});
            weights->vector(vblk(i, "ln2.weight"), {v_embd});
            weights->vector(vblk(i, "ln2.bias"), {v_embd});
            weights->vector(vblk(i, "ls1.weight"), {v_embd});
            weights->vector(vblk(i, "ls2.weight"), {v_embd});

            b.q_w = vblk(i, "attn_q.weight");
            b.k_w = vblk(i, "attn_k.weight");
            b.v_w = vblk(i, "attn_v.weight");
            b.out_w = vblk(i, "attn_out.weight");
            b.up_w = vblk(i, "ffn_up.weight");
            b.down_w = vblk(i, "ffn_down.weight");

            track_matrix(b.q_w, v_embd, v_embd);
            track_matrix(b.k_w, v_embd, v_embd);
            track_matrix(b.v_w, v_embd, v_embd);
            track_matrix(b.out_w, v_embd, v_embd);
            track_matrix(b.up_w, v_ff, v_embd);
            track_matrix(b.down_w, v_embd, v_ff);

            weights->vector(vblk(i, "attn_q.bias"), {v_embd});
            weights->vector(vblk(i, "attn_k.bias"), {v_embd});
            weights->vector(vblk(i, "attn_v.bias"), {v_embd});
            weights->vector(vblk(i, "attn_out.bias"), {v_embd});
            weights->vector(vblk(i, "ffn_up.bias"), {v_ff});
            weights->vector(vblk(i, "ffn_down.bias"), {v_embd});
        }
        track_matrix("mm.1.weight", v_ff, v_ff);
        track_matrix("mm.2.weight", v_proj_dim, v_ff);
        weights->finish();

        patch_embd = weights->read("v.patch_embd.weight");
        position_embd = weights->decode("v.position_embd.weight");
        class_embd = weights->read("v.class_embd");
        class_pos_embd = weights->read("v.class_position_embd.weight");
        pre_ln_w = weights->read("v.pre_ln.weight");
        pre_ln_b = weights->read("v.pre_ln.bias");

        for (std::size_t i = 0; i < v_layers; ++i) {
            auto& b = blocks[i];
            b.ln1_w = weights->read(vblk(i, "ln1.weight"));
            b.ln1_b = weights->read(vblk(i, "ln1.bias"));
            b.ln2_w = weights->read(vblk(i, "ln2.weight"));
            b.ln2_b = weights->read(vblk(i, "ln2.bias"));
            b.ls1_w = weights->read(vblk(i, "ls1.weight"));
            b.ls2_w = weights->read(vblk(i, "ls2.weight"));
            b.q_b = weights->read(vblk(i, "attn_q.bias"));
            b.k_b = weights->read(vblk(i, "attn_k.bias"));
            b.v_b = weights->read(vblk(i, "attn_v.bias"));
            b.out_b = weights->read(vblk(i, "attn_out.bias"));
            b.up_b = weights->read(vblk(i, "ffn_up.bias"));
            b.down_b = weights->read(vblk(i, "ffn_down.bias"));
        }
    }

    ImageSpan encode(std::span<const std::uint8_t> png_bytes, DenseCompute& compute) {
        weights->check_cancel();
        RgbImage src = decode_png_rgb(png_bytes);

        // Preserve aspect ratio downscale only if aligned size exceeds max pixels.
        std::size_t aligned_w = align_up(src.width, v_align);
        std::size_t aligned_h = align_up(src.height, v_align);
        if (aligned_w * aligned_h > v_max_pixels) {
            const double scale =
                std::sqrt(double(v_max_pixels) / double(src.width * src.height));
            const std::size_t new_w = std::max<std::size_t>(
                v_align,
                std::size_t(std::floor(double(src.width) * scale / double(v_align))) * v_align
            );
            const std::size_t new_h = std::max<std::size_t>(
                v_align,
                std::size_t(std::floor(double(src.height) * scale / double(v_align))) * v_align
            );
            src = resize_bilinear_rgb(src, new_w, new_h);
            aligned_w = align_up(src.width, v_align);
            aligned_h = align_up(src.height, v_align);
        }

        // Right/bottom pad with -1.0f (mean=0.5, std=0.5, black=0 -> -1.0).
        std::vector<float> pixels(aligned_w * aligned_h * 3, -1.0f);
        for (std::size_t y = 0; y < src.height; ++y) {
            for (std::size_t x = 0; x < src.width; ++x) {
                for (std::size_t c = 0; c < 3; ++c) {
                    const std::uint8_t u = src.rgb[(y * src.width + x) * 3 + c];
                    pixels[(y * aligned_w + x) * 3 + c] = 2.0f * float(u) / 255.0f - 1.0f;
                }
            }
        }

        const std::size_t grid_w = aligned_w / v_patch_size;
        const std::size_t grid_h = aligned_h / v_patch_size;
        const std::size_t n_patches = grid_w * grid_h;
        const std::size_t n_pos = n_patches + 1;

        // Patch embedding conv2d (20x20, stride 20) + resized 32x32 position embeddings.
        // Layout: row 0 = CLS token, rows 1..n_patches = patches in row-major order.
        Vec cur(n_pos * v_embd, 0.0f);
        for (std::size_t d = 0; d < v_embd; ++d)
            cur[d] = class_embd[d] + class_pos_embd[d];

        weights->parallel(n_patches, [&](std::size_t p) {
            const std::size_t py = p / grid_w;
            const std::size_t px = p % grid_w;
            float* out = cur.data() + (p + 1) * v_embd;

            // Bilinearly interpolated position embedding from 32x32 grid.
            const float sy = std::clamp(
                (float(py) + 0.5f) * (float(v_grid_base) / float(grid_h)) - 0.5f,
                0.0f,
                float(v_grid_base - 1)
            );
            const float sx = std::clamp(
                (float(px) + 0.5f) * (float(v_grid_base) / float(grid_w)) - 0.5f,
                0.0f,
                float(v_grid_base - 1)
            );
            const std::size_t y0 = std::size_t(std::floor(sy));
            const std::size_t y1 = std::min(y0 + 1, v_grid_base - 1);
            const std::size_t x0 = std::size_t(std::floor(sx));
            const std::size_t x1 = std::min(x0 + 1, v_grid_base - 1);
            const float wy = sy - float(y0);
            const float wx = sx - float(x0);

            const float* p00 = position_embd.data() + (y0 * v_grid_base + x0) * v_embd;
            const float* p01 = position_embd.data() + (y0 * v_grid_base + x1) * v_embd;
            const float* p10 = position_embd.data() + (y1 * v_grid_base + x0) * v_embd;
            const float* p11 = position_embd.data() + (y1 * v_grid_base + x1) * v_embd;

            // Patch conv2d: weight layout is [oc, c, ky, kx] (kx fastest).
            for (std::size_t oc = 0; oc < v_embd; ++oc) {
                float acc = 0.0f;
                for (std::size_t c = 0; c < 3; ++c) {
                    const float* kw =
                        patch_embd.data() + (oc * 3 + c) * (v_patch_size * v_patch_size);
                    for (std::size_t ky = 0; ky < v_patch_size; ++ky) {
                        const std::size_t iy = py * v_patch_size + ky;
                        const float* krow = kw + ky * v_patch_size;
                        for (std::size_t kx = 0; kx < v_patch_size; ++kx) {
                            const std::size_t ix = px * v_patch_size + kx;
                            acc += pixels[(iy * aligned_w + ix) * 3 + c] * krow[kx];
                        }
                    }
                }
                const float pos_val =
                    (1.0f - wy) * ((1.0f - wx) * p00[oc] + wx * p01[oc])
                    + wy * ((1.0f - wx) * p10[oc] + wx * p11[oc]);
                out[oc] = acc + pos_val;
            }
        });

        // CLS is at (0, 0); patches are 1-indexed (x+1, y+1).
        std::vector<int> pos_w;
        std::vector<int> pos_h;
        pos_w.reserve(n_pos);
        pos_h.reserve(n_pos);
        pos_w.push_back(0);
        pos_h.push_back(0);
        for (std::size_t y = 0; y < grid_h; ++y)
            for (std::size_t x = 0; x < grid_w; ++x) {
                pos_w.push_back(int(x + 1));
                pos_h.push_back(int(y + 1));
            }

        // Release temporary vision weights from device banks when leaving scope.
        struct ReleaseGuard {
            DenseCompute& compute;
            const std::vector<WeightIdentity>& ids;
            ~ReleaseGuard() {
                if (!ids.empty())
                    compute.release_weights(ids);
            }
        } release_guard{compute, matrix_identities};

        // Pre-LayerNorm
        layer_norm(cur, pre_ln_w, pre_ln_b, v_ln_eps);

        const float attn_scale = 1.0f / std::sqrt(float(v_head_dim));
        for (std::size_t i = 0; i < v_layers; ++i) {
            weights->check_cancel();
            const auto& b = blocks[i];

            Vec h = cur;
            layer_norm(h, b.ln1_w, b.ln1_b, v_ln_eps);

            const std::array<GgufWeights::Projection, 3> qkv_jobs = {
                {{b.q_w, h, 0}, {b.k_w, h, 0}, {b.v_w, h, 0}}
            };
            auto qkv = weights->project(compute, qkv_jobs);
            add_bias(qkv[0], b.q_b);
            add_bias(qkv[1], b.k_b);
            add_bias(qkv[2], b.v_b);

            apply_rope_2d(qkv[0], n_pos, pos_w, pos_h);
            apply_rope_2d(qkv[1], n_pos, pos_w, pos_h);

            // Bidirectional multi-head self-attention.
            Vec attn_out(n_pos * v_embd, 0.0f);
            weights->parallel(n_pos * v_heads, [&](std::size_t work) {
                const std::size_t r = work / v_heads;
                const std::size_t hd = work % v_heads;
                const float* qv = qkv[0].data() + (r * v_heads + hd) * v_head_dim;
                std::vector<float> scores(n_pos);
                float max_score = -INFINITY;
                for (std::size_t t = 0; t < n_pos; ++t) {
                    const float* kv = qkv[1].data() + (t * v_heads + hd) * v_head_dim;
                    float dot = 0.0f;
                    for (std::size_t d = 0; d < v_head_dim; ++d)
                        dot += qv[d] * kv[d];
                    const float s = dot * attn_scale;
                    scores[t] = s;
                    max_score = std::max(max_score, s);
                }
                double denom = 0.0;
                for (std::size_t t = 0; t < n_pos; ++t) {
                    scores[t] = std::exp(scores[t] - max_score);
                    denom += double(scores[t]);
                }
                float* dst = attn_out.data() + (r * v_heads + hd) * v_head_dim;
                for (std::size_t t = 0; t < n_pos; ++t) {
                    const float p = float(double(scores[t]) / denom);
                    const float* vv = qkv[2].data() + (t * v_heads + hd) * v_head_dim;
                    for (std::size_t d = 0; d < v_head_dim; ++d)
                        dst[d] += p * vv[d];
                }
            });

            Vec proj_out = weights->project(compute, b.out_w, attn_out);
            add_bias(proj_out, b.out_b);
            for (std::size_t p = 0; p < n_pos; ++p)
                for (std::size_t d = 0; d < v_embd; ++d)
                    cur[p * v_embd + d] += proj_out[p * v_embd + d] * b.ls1_w[d];

            Vec ffn_in = cur;
            layer_norm(ffn_in, b.ln2_w, b.ln2_b, v_ln_eps);
            Vec up = weights->project(compute, b.up_w, ffn_in);
            add_bias(up, b.up_b);
            for (auto& v : up)
                v = gelu_erf(v);
            Vec down = weights->project(compute, b.down_w, up);
            add_bias(down, b.down_b);
            for (std::size_t p = 0; p < n_pos; ++p)
                for (std::size_t d = 0; d < v_embd; ++d)
                    cur[p * v_embd + d] += down[p * v_embd + d] * b.ls2_w[d];
        }

        // Drop CLS (row 0) and apply 2x2 spatial merge:
        // merged(by, bx) = [patch(2*by, 2*bx), patch(2*by, 2*bx+1),
        //                   patch(2*by+1, 2*bx), patch(2*by+1, 2*bx+1)]
        const std::size_t merged_w = grid_w / v_merge;
        const std::size_t merged_h = grid_h / v_merge;
        const std::size_t n_tokens = merged_w * merged_h;
        Vec merged(n_tokens * (v_embd * v_merge * v_merge));
        for (std::size_t by = 0; by < merged_h; ++by) {
            for (std::size_t bx = 0; bx < merged_w; ++bx) {
                float* dst = merged.data() + (by * merged_w + bx) * (4 * v_embd);
                const std::size_t p00 = (2 * by) * grid_w + (2 * bx) + 1;
                const std::size_t p01 = (2 * by) * grid_w + (2 * bx + 1) + 1;
                const std::size_t p10 = (2 * by + 1) * grid_w + (2 * bx) + 1;
                const std::size_t p11 = (2 * by + 1) * grid_w + (2 * bx + 1) + 1;
                std::memcpy(dst + 0 * v_embd, cur.data() + p00 * v_embd, v_embd * sizeof(float));
                std::memcpy(dst + 1 * v_embd, cur.data() + p01 * v_embd, v_embd * sizeof(float));
                std::memcpy(dst + 2 * v_embd, cur.data() + p10 * v_embd, v_embd * sizeof(float));
                std::memcpy(dst + 3 * v_embd, cur.data() + p11 * v_embd, v_embd * sizeof(float));
            }
        }

        // 2-layer bias-free MLP projector: mm.1 -> GELU(erf) -> mm.2
        Vec mm1 = weights->project(compute, "mm.1.weight", merged);
        for (auto& v : mm1)
            v = gelu_erf(v);
        Vec projected = weights->project(compute, "mm.2.weight", mm1);
        need(projected.size() == n_tokens * v_proj_dim, "inkling2v_projected_shape");

        ImageSpan span;
        span.tokens = n_tokens;
        span.embeddings = std::move(projected);
        return span;
    }
};

std::string InklingVision::find_mmproj(const char* root) {
    if (!root || !*root)
        return {};
    std::string found;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_regular_file())
            continue;
        const auto name = entry.path().filename().string();
        if (name.starts_with("mmproj-") && name.ends_with(".gguf")) {
            need(found.empty(), "inkling2v_multiple_mmproj");
            found = entry.path().string();
        }
    }
    return found;
}

InklingVision::InklingVision(
    const std::string& mmproj_path, const std::atomic_bool* cancel
)
    : impl_(std::make_unique<Impl>(mmproj_path, cancel)) {
}

InklingVision::~InklingVision() = default;

void InklingVision::set_cancel(const std::atomic_bool& cancel) {
    impl_->weights->set_cancel(cancel);
}

ImageSpan
InklingVision::encode(std::span<const std::uint8_t> png_bytes, DenseCompute& compute) {
    return impl_->encode(png_bytes, compute);
}

std::size_t image_token_count(std::span<const std::uint8_t> png_bytes) {
    PngReader reader(png_bytes);
    const auto& image = reader.image;
    need(
        image.width >= 1 && image.height >= 1 && image.width <= 16384
            && image.height <= 16384,
        "inkling2v_png_dimensions"
    );
    std::size_t aligned_w = align_up(image.width, v_align);
    std::size_t aligned_h = align_up(image.height, v_align);
    if (aligned_w * aligned_h > v_max_pixels) {
        const double scale =
            std::sqrt(double(v_max_pixels) / double(std::size_t(image.width) * image.height));
        aligned_w = std::max<std::size_t>(
            v_align,
            std::size_t(std::floor(double(image.width) * scale / double(v_align))) * v_align
        );
        aligned_h = std::max<std::size_t>(
            v_align,
            std::size_t(std::floor(double(image.height) * scale / double(v_align))) * v_align
        );
    }
    return (aligned_w / v_align) * (aligned_h / v_align);
}
}  // namespace kadan::text
