#include "kadan/lfm3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <png.h>
#include <stdexcept>

#include "kadan/float_weights.hpp"

namespace kadan::lfm3 {
namespace {
void check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

void check_cancel(const std::atomic_bool& cancel) {
    if (cancel.load(std::memory_order_relaxed))
        throw std::runtime_error("lfm3_vision_cancelled");
}

double gelu_f64(double x) {
    return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
}

double silu_f64(double x) {
    const double e = std::exp(-std::abs(x));
    const double sig = x >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
    return x * sig;
}

// ---------------------------------------------------------------------------
// Minimal bounds-checked baseline JPEG (SOF0) decoder — zero external deps.
// ---------------------------------------------------------------------------
struct JpegHuffTable {
    std::array<int, 17> maxcode{};
    std::array<int, 17> valoffset{};
    std::array<std::uint8_t, 256> huffval{};
    bool valid = false;

    void build(const std::uint8_t* bits, const std::uint8_t* vals, int count) {
        std::array<int, 257> huffsize{};
        std::array<int, 257> huffcode{};
        int k = 0;
        for (int i = 1; i <= 16; ++i) {
            for (int j = 0; j < bits[i - 1]; ++j) {
                check(k < 256, "jpeg_huffman_overflow");
                huffsize[k++] = i;
            }
        }
        check(k == count, "jpeg_huffman_count");
        std::memcpy(huffval.data(), vals, static_cast<std::size_t>(count));
        int code = 0;
        int si = huffsize[0];
        int p = 0;
        while (p < k) {
            while (p < k && huffsize[p] == si) {
                huffcode[p++] = code++;
            }
            code <<= 1;
            ++si;
        }
        p = 0;
        for (int l = 1; l <= 16; ++l) {
            if (bits[l - 1] != 0) {
                valoffset[l] = p - huffcode[p];
                p += bits[l - 1];
                maxcode[l] = huffcode[p - 1];
            } else {
                maxcode[l] = -1;
            }
        }
        valid = true;
    }
};

struct JpegBitReader {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t pos = 0;
    std::uint32_t bit_buf = 0;
    int bit_count = 0;

    std::uint8_t next_byte() {
        check(pos < size, "jpeg_truncated_scan");
        std::uint8_t b = data[pos++];
        if (b == 0xFF) {
            check(pos < size, "jpeg_truncated_marker");
            std::uint8_t next = data[pos++];
            while (next == 0xFF) {
                check(pos < size, "jpeg_truncated_marker");
                next = data[pos++];
            }
            if (next == 0x00)
                return 0xFF;
            // Restart marker or EOI inside scan.
            pos -= 2;
            return 0;
        }
        return b;
    }

    int get_bits(int n) {
        while (bit_count < n) {
            bit_buf = (bit_buf << 8) | next_byte();
            bit_count += 8;
        }
        bit_count -= n;
        return static_cast<int>((bit_buf >> bit_count) & ((1U << n) - 1U));
    }

    int decode_huff(const JpegHuffTable& t) {
        check(t.valid, "jpeg_missing_huffman");
        int code = 0;
        for (int l = 1; l <= 16; ++l) {
            code = (code << 1) | get_bits(1);
            if (t.maxcode[l] >= 0 && code <= t.maxcode[l])
                return t.huffval[static_cast<std::size_t>(code + t.valoffset[l])];
        }
        throw std::runtime_error("jpeg_invalid_huffman_code");
    }

    int receive_extend(int s) {
        if (s == 0)
            return 0;
        int v = get_bits(s);
        if (v < (1 << (s - 1)))
            v += (-1 << s) + 1;
        return v;
    }
};

constexpr std::array<std::uint8_t, 64> kZigZag = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};

void idct_8x8(const std::array<float, 64>& in, std::array<std::uint8_t, 64>& out) {
    static const auto cos_table = [] {
        std::array<std::array<float, 8>, 8> t{};
        constexpr double pi = 3.14159265358979323846;
        for (int u = 0; u < 8; ++u)
            for (int x = 0; x < 8; ++x)
                t[u][x] = static_cast<float>(
                    std::cos((2.0 * x + 1.0) * u * pi / 16.0)
                    * (u == 0 ? 0.35355339059327376 : 0.5)
                );
        return t;
    }();
    std::array<float, 64> tmp{};
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            float s = 0.0f;
            for (int u = 0; u < 8; ++u)
                s += in[static_cast<std::size_t>(y * 8 + u)] * cos_table[u][x];
            tmp[static_cast<std::size_t>(y * 8 + x)] = s;
        }
    }
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            float s = 0.0f;
            for (int v = 0; v < 8; ++v)
                s += tmp[static_cast<std::size_t>(v * 8 + x)] * cos_table[v][y];
            const int val = static_cast<int>(std::lround(s + 128.0f));
            out[static_cast<std::size_t>(y * 8 + x)] =
                static_cast<std::uint8_t>(std::clamp(val, 0, 255));
        }
    }
}

RgbImage decode_baseline_jpeg(std::string_view raw, std::size_t max_pixels) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(raw.data());
    const std::size_t len = raw.size();
    check(len >= 4 && bytes[0] == 0xFF && bytes[1] == 0xD8, "jpeg_invalid_header");

    std::array<std::array<std::uint16_t, 64>, 4> qtables{};
    std::array<bool, 4> qvalid{};
    std::array<JpegHuffTable, 4> dc_tables{};
    std::array<JpegHuffTable, 4> ac_tables{};
    struct Comp {
        int id = 0, hsamp = 1, vsamp = 1, qid = 0, dc = 0, ac = 0, pred = 0;
    };
    std::vector<Comp> comps;
    std::size_t width = 0, height = 0;
    int max_h = 1, max_v = 1;
    std::uint16_t restart_interval = 0;
    std::size_t pos = 2;
    std::size_t scan_start = 0;

    auto read_u16 = [&](std::size_t p) -> std::uint16_t {
        check(p + 2 <= len, "jpeg_truncated");
        return static_cast<std::uint16_t>((bytes[p] << 8) | bytes[p + 1]);
    };

    while (pos < len) {
        check(bytes[pos] == 0xFF, "jpeg_marker_sync");
        while (pos < len && bytes[pos] == 0xFF)
            ++pos;
        check(pos < len, "jpeg_truncated_marker");
        const std::uint8_t marker = bytes[pos++];
        if (marker == 0xD9)
            break;
        if (marker >= 0xD0 && marker <= 0xD7)
            continue;
        const std::uint16_t seg_len = read_u16(pos);
        check(seg_len >= 2 && pos + seg_len <= len, "jpeg_bad_segment");
        const std::size_t seg_data = pos + 2;
        const std::size_t seg_end = pos + seg_len;
        if (marker == 0xDB) {  // DQT
            std::size_t p = seg_data;
            while (p < seg_end) {
                const std::uint8_t pq_tq = bytes[p++];
                const int pq = pq_tq >> 4;
                const int tq = pq_tq & 0x0F;
                check(tq < 4 && (pq == 0 || pq == 1), "jpeg_bad_dqt");
                if (pq == 0) {
                    check(p + 64 <= seg_end, "jpeg_truncated_dqt");
                    for (int i = 0; i < 64; ++i)
                        qtables[tq][i] = bytes[p++];
                } else {
                    check(p + 128 <= seg_end, "jpeg_truncated_dqt");
                    for (int i = 0; i < 64; ++i) {
                        qtables[tq][i] = read_u16(p);
                        p += 2;
                    }
                }
                qvalid[tq] = true;
            }
        } else if (marker == 0xC4) {  // DHT
            std::size_t p = seg_data;
            while (p < seg_end) {
                const std::uint8_t tc_th = bytes[p++];
                const int tc = tc_th >> 4;
                const int th = tc_th & 0x0F;
                check((tc == 0 || tc == 1) && th < 4 && p + 16 <= seg_end, "jpeg_bad_dht");
                const std::uint8_t* bits = bytes + p;
                p += 16;
                int count = 0;
                for (int i = 0; i < 16; ++i)
                    count += bits[i];
                check(count <= 256 && p + static_cast<std::size_t>(count) <= seg_end, "jpeg_dht_len");
                (tc == 0 ? dc_tables[th] : ac_tables[th]).build(bits, bytes + p, count);
                p += static_cast<std::size_t>(count);
            }
        } else if (marker == 0xDD) {  // DRI
            check(seg_len == 4, "jpeg_bad_dri");
            restart_interval = read_u16(seg_data);
        } else if (marker == 0xC0) {  // SOF0 baseline
            check(seg_len >= 8, "jpeg_bad_sof0");
            const int precision = bytes[seg_data];
            check(precision == 8, "jpeg_only_8bit_supported");
            height = read_u16(seg_data + 1);
            width = read_u16(seg_data + 3);
            const int ncomp = bytes[seg_data + 5];
            check(
                width > 0 && height > 0 && width * height <= max_pixels
                    && (ncomp == 1 || ncomp == 3),
                "jpeg_invalid_dimensions"
            );
            check(seg_data + 6 + static_cast<std::size_t>(ncomp) * 3 == seg_end, "jpeg_sof0_len");
            comps.resize(static_cast<std::size_t>(ncomp));
            max_h = max_v = 1;
            for (int i = 0; i < ncomp; ++i) {
                const std::size_t off = seg_data + 6 + static_cast<std::size_t>(i) * 3;
                comps[i].id = bytes[off];
                comps[i].hsamp = bytes[off + 1] >> 4;
                comps[i].vsamp = bytes[off + 1] & 0x0F;
                comps[i].qid = bytes[off + 2];
                check(
                    comps[i].hsamp >= 1 && comps[i].hsamp <= 2 && comps[i].vsamp >= 1
                        && comps[i].vsamp <= 2 && comps[i].qid < 4,
                    "jpeg_unsupported_sampling"
                );
                max_h = std::max(max_h, comps[i].hsamp);
                max_v = std::max(max_v, comps[i].vsamp);
            }
        } else if (marker == 0xC2) {
            throw std::runtime_error("jpeg_progressive_not_supported");
        } else if (marker == 0xDA) {  // SOS
            check(!comps.empty(), "jpeg_sos_before_sof");
            const int ns = bytes[seg_data];
            check(ns == static_cast<int>(comps.size()), "jpeg_sos_components");
            for (int i = 0; i < ns; ++i) {
                const int cid = bytes[seg_data + 1 + static_cast<std::size_t>(i) * 2];
                const std::uint8_t td_ta =
                    bytes[seg_data + 2 + static_cast<std::size_t>(i) * 2];
                bool matched = false;
                for (auto& c : comps) {
                    if (c.id == cid) {
                        c.dc = td_ta >> 4;
                        c.ac = td_ta & 0x0F;
                        check(c.dc < 4 && c.ac < 4, "jpeg_sos_table_id");
                        matched = true;
                        break;
                    }
                }
                check(matched, "jpeg_sos_unknown_component");
            }
            scan_start = seg_end;
            break;
        }
        pos = seg_end;
    }

    check(scan_start > 0 && width > 0 && height > 0, "jpeg_missing_scan");
    for (const auto& c : comps)
        check(qvalid[c.qid], "jpeg_missing_dqt");

    const std::size_t mcu_w = static_cast<std::size_t>(max_h * 8);
    const std::size_t mcu_h = static_cast<std::size_t>(max_v * 8);
    const std::size_t mcus_x = (width + mcu_w - 1) / mcu_w;
    const std::size_t mcus_y = (height + mcu_h - 1) / mcu_h;

    RgbImage out;
    out.width = width;
    out.height = height;
    out.rgb.assign(width * height * 3, 0);

    JpegBitReader reader{bytes, len, scan_start, 0, 0};
    std::size_t mcu_count = 0;

    for (std::size_t my = 0; my < mcus_y; ++my) {
        for (std::size_t mx = 0; mx < mcus_x; ++mx) {
            if (restart_interval > 0 && mcu_count > 0
                && (mcu_count % restart_interval) == 0) {
                reader.bit_buf = 0;
                reader.bit_count = 0;
                while (reader.pos < len && bytes[reader.pos] != 0xFF)
                    ++reader.pos;
                if (reader.pos + 1 < len && bytes[reader.pos] == 0xFF
                    && bytes[reader.pos + 1] >= 0xD0 && bytes[reader.pos + 1] <= 0xD7) {
                    reader.pos += 2;
                }
                for (auto& c : comps)
                    c.pred = 0;
            }
            ++mcu_count;

            std::array<std::vector<std::uint8_t>, 3> plane;
            for (std::size_t ci = 0; ci < comps.size(); ++ci) {
                auto& c = comps[ci];
                const std::size_t pw = static_cast<std::size_t>(c.hsamp * 8);
                const std::size_t ph = static_cast<std::size_t>(c.vsamp * 8);
                plane[ci].assign(pw * ph, 0);
                for (int bv = 0; bv < c.vsamp; ++bv) {
                    for (int bh = 0; bh < c.hsamp; ++bh) {
                        std::array<float, 64> coeff{};
                        const int s = reader.decode_huff(dc_tables[c.dc]);
                        const int diff = reader.receive_extend(s);
                        c.pred += diff;
                        coeff[0] = static_cast<float>(c.pred * qtables[c.qid][0]);
                        int k = 1;
                        while (k < 64) {
                            const int rs = reader.decode_huff(ac_tables[c.ac]);
                            if (rs == 0)
                                break;
                            const int run = rs >> 4;
                            const int sz = rs & 0x0F;
                            if (rs == 0xF0) {
                                k += 16;
                                continue;
                            }
                            k += run;
                            check(k < 64, "jpeg_ac_overflow");
                            const int ac_val = reader.receive_extend(sz);
                            coeff[kZigZag[k]] =
                                static_cast<float>(ac_val * qtables[c.qid][k]);
                            ++k;
                        }
                        std::array<std::uint8_t, 64> block{};
                        idct_8x8(coeff, block);
                        for (int y = 0; y < 8; ++y) {
                            for (int x = 0; x < 8; ++x) {
                                plane[ci][(static_cast<std::size_t>(bv * 8 + y)) * pw
                                          + static_cast<std::size_t>(bh * 8 + x)] =
                                    block[static_cast<std::size_t>(y * 8 + x)];
                            }
                        }
                    }
                }
            }

            for (std::size_t y = 0; y < mcu_h; ++y) {
                const std::size_t iy = my * mcu_h + y;
                if (iy >= height)
                    break;
                for (std::size_t x = 0; x < mcu_w; ++x) {
                    const std::size_t ix = mx * mcu_w + x;
                    if (ix >= width)
                        break;
                    auto sample_comp = [&](std::size_t ci) -> float {
                        const auto& c = comps[ci];
                        const std::size_t cx = x * static_cast<std::size_t>(c.hsamp)
                                               / static_cast<std::size_t>(max_h);
                        const std::size_t cy = y * static_cast<std::size_t>(c.vsamp)
                                               / static_cast<std::size_t>(max_v);
                        return static_cast<float>(
                            plane[ci][cy * static_cast<std::size_t>(c.hsamp * 8) + cx]
                        );
                    };
                    std::uint8_t* dst = out.rgb.data() + (iy * width + ix) * 3;
                    if (comps.size() == 1) {
                        const auto g = static_cast<std::uint8_t>(
                            std::clamp(static_cast<int>(std::lround(sample_comp(0))), 0, 255)
                        );
                        dst[0] = dst[1] = dst[2] = g;
                    } else {
                        const float Y = sample_comp(0);
                        const float Cb = sample_comp(1) - 128.0f;
                        const float Cr = sample_comp(2) - 128.0f;
                        dst[0] = static_cast<std::uint8_t>(std::clamp(
                            static_cast<int>(std::lround(Y + 1.402f * Cr)), 0, 255
                        ));
                        dst[1] = static_cast<std::uint8_t>(std::clamp(
                            static_cast<int>(
                                std::lround(Y - 0.344136f * Cb - 0.714136f * Cr)
                            ),
                            0,
                            255
                        ));
                        dst[2] = static_cast<std::uint8_t>(std::clamp(
                            static_cast<int>(std::lround(Y + 1.772f * Cb)), 0, 255
                        ));
                    }
                }
            }
        }
    }
    return out;
}

float cubic_weight(float t) {
    t = std::abs(t);
    constexpr float a = -0.5f;
    if (t < 1.0f)
        return ((a + 2.0f) * t - (a + 3.0f)) * t * t + 1.0f;
    if (t < 2.0f)
        return ((a * t - 5.0f * a) * t + 8.0f * a) * t - 4.0f * a;
    return 0.0f;
}

std::vector<float>
bicubic_resize_rgb(const RgbImage& src, std::size_t dst_h, std::size_t dst_w) {
    // Returns float RGB in [0, 1], layout [dst_h, dst_w, 3].
    std::vector<float> tmp(src.height * dst_w * 3, 0.0f);
    const float scale_x = static_cast<float>(src.width) / static_cast<float>(dst_w);
    for (std::size_t x = 0; x < dst_w; ++x) {
        const float cx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
        const int ix = static_cast<int>(std::floor(cx));
        float ws[4];
        int xs[4];
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) {
            xs[k] = std::clamp(ix - 1 + k, 0, static_cast<int>(src.width) - 1);
            ws[k] = cubic_weight(cx - static_cast<float>(ix - 1 + k));
            sum += ws[k];
        }
        for (float& w : ws)
            w /= sum;
        for (std::size_t y = 0; y < src.height; ++y) {
            for (int c = 0; c < 3; ++c) {
                float v = 0.0f;
                for (int k = 0; k < 4; ++k)
                    v += ws[k]
                         * (static_cast<float>(
                                src.rgb[(y * src.width + static_cast<std::size_t>(xs[k])) * 3 + c]
                            )
                            / 255.0f);
                tmp[(y * dst_w + x) * 3 + c] = v;
            }
        }
    }
    std::vector<float> dst(dst_h * dst_w * 3, 0.0f);
    const float scale_y = static_cast<float>(src.height) / static_cast<float>(dst_h);
    for (std::size_t y = 0; y < dst_h; ++y) {
        const float cy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int iy = static_cast<int>(std::floor(cy));
        float ws[4];
        int ys[4];
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) {
            ys[k] = std::clamp(iy - 1 + k, 0, static_cast<int>(src.height) - 1);
            ws[k] = cubic_weight(cy - static_cast<float>(iy - 1 + k));
            sum += ws[k];
        }
        for (float& w : ws)
            w /= sum;
        for (std::size_t x = 0; x < dst_w; ++x) {
            for (int c = 0; c < 3; ++c) {
                float v = 0.0f;
                for (int k = 0; k < 4; ++k)
                    v += ws[k] * tmp[(static_cast<std::size_t>(ys[k]) * dst_w + x) * 3 + c];
                dst[(y * dst_w + x) * 3 + c] = std::clamp(v, 0.0f, 1.0f);
            }
        }
    }
    return dst;
}
}  // namespace

VisionConfig VisionConfig::parse(const Json& root) {
    const Json& v = root.at("vision_config");
    const Json& t = root.at("text_config");
    VisionConfig c;
    c.hidden_size = v.value("hidden_size", c.hidden_size);
    c.intermediate_size = v.value("intermediate_size", c.intermediate_size);
    c.num_hidden_layers = v.value("num_hidden_layers", c.num_hidden_layers);
    c.num_attention_heads = v.value("num_attention_heads", c.num_attention_heads);
    c.num_key_value_heads = v.value("num_key_value_heads", c.num_key_value_heads);
    c.patch_size = v.value("patch_size", c.patch_size);
    c.temporal_patch_size = v.value("temporal_patch_size", c.temporal_patch_size);
    c.spatial_merge_size = v.value("spatial_merge_size", c.spatial_merge_size);
    c.conv_kernel_size = v.value("conv_kernel_size", c.conv_kernel_size);
    c.in_channels = v.value("in_channels", c.in_channels);
    c.norm_eps = v.value("norm_eps", c.norm_eps);
    if (v.contains("rope_parameters") && v.at("rope_parameters").is_object())
        c.rope_theta = v.at("rope_parameters").value("rope_theta", c.rope_theta);
    c.downsample_factor = root.value("downsample_factor", c.downsample_factor);
    c.projector_hidden_size = root.value("projector_hidden_size", c.projector_hidden_size);
    c.projector_layer_norm_eps =
        root.value("projector_layer_norm_eps", c.projector_layer_norm_eps);
    c.projector_bias = root.value("projector_bias", c.projector_bias);
    c.image_token_id = root.value("image_token_id", c.image_token_id);
    c.text_hidden_size = t.value("hidden_size", c.text_hidden_size);
    c.layer_types.clear();
    for (const auto& entry : v.at("layer_types")) {
        const auto s = entry.get<std::string>();
        c.layer_types.push_back(
            s == "full_attention" ? LayerKind::full_attention : LayerKind::conv
        );
    }
    check(c.layer_types.size() == c.num_hidden_layers, "lfm3_vl_layer_types");
    return c;
}

std::pair<std::size_t, std::size_t> smart_resize_lfm3_vl(
    std::size_t height,
    std::size_t width,
    std::size_t factor,
    std::size_t min_pixels,
    std::size_t max_pixels
) {
    check(height > 0 && width > 0 && factor > 0, "lfm3_vl_invalid_dimensions");
    const double aspect =
        static_cast<double>(std::max(height, width))
        / static_cast<double>(std::min(height, width));
    check(aspect <= 200.0, "lfm3_vl_aspect_ratio_exceeded");
    const auto round_to = [factor](double v) -> std::size_t {
        return std::max<std::size_t>(
            factor, static_cast<std::size_t>(std::llround(v / factor)) * factor
        );
    };
    std::size_t h_bar = round_to(static_cast<double>(height));
    std::size_t w_bar = round_to(static_cast<double>(width));
    if (h_bar * w_bar > max_pixels) {
        const double beta =
            std::sqrt(static_cast<double>(height * width) / static_cast<double>(max_pixels));
        h_bar = std::max<std::size_t>(
            factor,
            static_cast<std::size_t>(std::floor(static_cast<double>(height) / beta / factor))
                * factor
        );
        w_bar = std::max<std::size_t>(
            factor,
            static_cast<std::size_t>(std::floor(static_cast<double>(width) / beta / factor))
                * factor
        );
    } else if (h_bar * w_bar < min_pixels) {
        const double beta =
            std::sqrt(static_cast<double>(min_pixels) / static_cast<double>(height * width));
        h_bar =
            static_cast<std::size_t>(std::ceil(static_cast<double>(height) * beta / factor))
            * factor;
        w_bar =
            static_cast<std::size_t>(std::ceil(static_cast<double>(width) * beta / factor))
            * factor;
    }
    return {h_bar, w_bar};
}

RgbImage decode_vision_image(
    std::string_view raw_bytes, std::string_view media_type, std::size_t max_pixels
) {
    check(!raw_bytes.empty(), "lfm3_vl_empty_image");
    const bool is_png = raw_bytes.size() >= 8
                        && static_cast<unsigned char>(raw_bytes[0]) == 0x89
                        && raw_bytes.substr(1, 3) == "PNG";
    const bool is_jpeg = raw_bytes.size() >= 3
                         && static_cast<unsigned char>(raw_bytes[0]) == 0xFF
                         && static_cast<unsigned char>(raw_bytes[1]) == 0xD8
                         && static_cast<unsigned char>(raw_bytes[2]) == 0xFF;
    if (media_type == "image/png") {
        check(is_png, "Screenshot format must match its media type and fit 4,194,304 pixels.");
        png_image img{};
        img.version = PNG_IMAGE_VERSION;
        check(
            png_image_begin_read_from_memory(&img, raw_bytes.data(), raw_bytes.size()) != 0,
            "Invalid PNG image"
        );
        struct Guard {
            png_image& i;
            ~Guard() {
                png_image_free(&i);
            }
        } guard{img};
        check(
            img.width > 0 && img.height > 0
                && std::size_t(img.width) * img.height <= max_pixels,
            "Screenshot format must match its media type and fit 4,194,304 pixels."
        );
        img.format = PNG_FORMAT_RGB;
        RgbImage out;
        out.width = img.width;
        out.height = img.height;
        out.rgb.resize(out.width * out.height * 3);
        check(
            png_image_finish_read(&img, nullptr, out.rgb.data(), 0, nullptr) != 0,
            "Invalid PNG pixels"
        );
        return out;
    }
    if (media_type == "image/jpeg") {
        check(is_jpeg, "Screenshot format must match its media type and fit 4,194,304 pixels.");
        return decode_baseline_jpeg(raw_bytes, max_pixels);
    }
    throw std::runtime_error("Screenshot must be PNG or JPEG.");
}

VisionPatches preprocess_vision_image(const RgbImage& image, const VisionConfig& config) {
    const std::size_t factor = config.patch_size * config.spatial_merge_size;
    const auto [resized_h, resized_w] = smart_resize_lfm3_vl(
        image.height, image.width, factor, 65536, 16777216
    );
    auto rgb = bicubic_resize_rgb(image, resized_h, resized_w);
    // Normalize with mean=0.5, std=0.5 -> [-1, 1].
    for (float& v : rgb)
        v = (v - 0.5f) / 0.5f;

    const std::size_t grid_h = resized_h / config.patch_size;
    const std::size_t grid_w = resized_w / config.patch_size;
    const std::size_t merge = config.spatial_merge_size;
    const std::size_t ps = config.patch_size;
    const std::size_t tp = config.temporal_patch_size;
    const std::size_t patch_dim = config.in_channels * tp * ps * ps;

    VisionPatches out;
    out.grid_h = grid_h;
    out.grid_w = grid_w;
    out.num_patches = grid_h * grid_w;
    out.patch_dim = patch_dim;
    out.pixel_values.resize(out.num_patches * patch_dim);

    // Order matches image_processing_lfm3_vl.py patchify():
    // [grid_h/merge, grid_w/merge, merge, merge, channel, temporal_patch, patch_y, patch_x]
    std::size_t patch_idx = 0;
    for (std::size_t bh = 0; bh < grid_h / merge; ++bh) {
        for (std::size_t bw = 0; bw < grid_w / merge; ++bw) {
            for (std::size_t mh = 0; mh < merge; ++mh) {
                for (std::size_t mw = 0; mw < merge; ++mw) {
                    const std::size_t py0 = (bh * merge + mh) * ps;
                    const std::size_t px0 = (bw * merge + mw) * ps;
                    float* dst = out.pixel_values.data() + patch_idx * patch_dim;
                    std::size_t off = 0;
                    for (std::size_t c = 0; c < config.in_channels; ++c) {
                        for (std::size_t t = 0; t < tp; ++t) {
                            for (std::size_t dy = 0; dy < ps; ++dy) {
                                for (std::size_t dx = 0; dx < ps; ++dx) {
                                    dst[off++] = rgb[((py0 + dy) * resized_w + (px0 + dx)) * 3 + c];
                                }
                            }
                        }
                    }
                    ++patch_idx;
                }
            }
        }
    }
    return out;
}

VisionEncoder::VisionEncoder(
    VisionConfig config,
    std::shared_ptr<checkpoint::Shard> shard,
    std::shared_ptr<Resources> resources,
    std::shared_ptr<DenseCompute> compute
)
    : config_(std::move(config)), shard_(std::move(shard)),
      resources_(std::move(resources)), compute_(std::move(compute)) {
    check(bool(shard_) && bool(resources_), "lfm3_vl_encoder_deps");
}

std::vector<float> VisionEncoder::encode(
    const VisionPatches& patches, const std::atomic_bool& cancel
) const {
    const std::size_t N = patches.num_patches;
    const std::size_t H = config_.hidden_size;
    const std::size_t head_dim = H / config_.num_attention_heads;
    const std::size_t kv_dim = config_.num_key_value_heads * head_dim;
    const std::size_t merge = config_.spatial_merge_size;

    // 1. Patchifier linear + embedding RMSNorm
    std::vector<float> hidden(N * H);
    checkpoint::project_source(
        *resources_,
        Workload::lfm3_vision,
        compute_.get(),
        *shard_,
        "model.vision_tower.patchifier.weight",
        {},
        patches.pixel_values,
        patches.patch_dim,
        H,
        hidden,
        cancel
    );
    {
        auto norm_w = checkpoint::float_slice(
            *resources_,
            Workload::lfm3_vision,
            *shard_,
            "model.vision_tower.embedding_norm.weight",
            0,
            H,
            cancel
        );
        rms_norm_lfm3(hidden, norm_w, N, H, config_.norm_eps, hidden);
    }

    // 2. Axial 2D RoPE cos/sin tables for merge-block-ordered patches.
    const std::size_t spatial_dim = head_dim / 2;
    const std::size_t axis_freqs = spatial_dim / 2;
    std::vector<float> inv_freq(axis_freqs);
    for (std::size_t i = 0; i < axis_freqs; ++i) {
        inv_freq[i] = static_cast<float>(
            1.0 / std::pow(config_.rope_theta, static_cast<double>(2 * i) / spatial_dim)
        );
    }
    std::vector<float> cos_tab(N * head_dim);
    std::vector<float> sin_tab(N * head_dim);
    {
        std::size_t idx = 0;
        for (std::size_t bh = 0; bh < patches.grid_h / merge; ++bh) {
            for (std::size_t bw = 0; bw < patches.grid_w / merge; ++bw) {
                for (std::size_t mh = 0; mh < merge; ++mh) {
                    for (std::size_t mw = 0; mw < merge; ++mw) {
                        const float ph = static_cast<float>(bh * merge + mh);
                        const float pw = static_cast<float>(bw * merge + mw);
                        float* cp = cos_tab.data() + idx * head_dim;
                        float* sp = sin_tab.data() + idx * head_dim;
                        for (std::size_t i = 0; i < axis_freqs; ++i) {
                            const float ah = ph * inv_freq[i];
                            const float aw = pw * inv_freq[i];
                            cp[i] = cp[i + spatial_dim] = std::cos(ah);
                            sp[i] = sp[i + spatial_dim] = std::sin(ah);
                            cp[axis_freqs + i] = cp[spatial_dim + axis_freqs + i] =
                                std::cos(aw);
                            sp[axis_freqs + i] = sp[spatial_dim + axis_freqs + i] =
                                std::sin(aw);
                        }
                        ++idx;
                    }
                }
            }
        }
    }

    auto apply_axial_rope = [&](std::vector<float>& t, std::size_t heads) {
        const std::size_t half = head_dim / 2;
        for (std::size_t p = 0; p < N; ++p) {
            const float* cp = cos_tab.data() + p * head_dim;
            const float* sp = sin_tab.data() + p * head_dim;
            for (std::size_t h = 0; h < heads; ++h) {
                float* v = t.data() + (p * heads + h) * head_dim;
                for (std::size_t d = 0; d < half; ++d) {
                    const float x1 = v[d];
                    const float x2 = v[d + half];
                    v[d] = x1 * cp[d] - x2 * sp[d];
                    v[d + half] = x2 * cp[d + half] + x1 * sp[d + half];
                }
            }
        }
    };

    std::vector<float> normed(N * H);
    std::vector<float> op_out(N * H);
    std::vector<float> ffn_gate(N * config_.intermediate_size);
    std::vector<float> ffn_up(N * config_.intermediate_size);

    for (std::size_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
        check_cancel(cancel);
        const std::string lp =
            "model.vision_tower.layers." + std::to_string(layer) + ".";
        auto op_nw = checkpoint::float_slice(
            *resources_, Workload::lfm3_vision, *shard_, lp + "operator_norm.weight", 0, H, cancel
        );
        rms_norm_lfm3(hidden, op_nw, N, H, config_.norm_eps, normed);

        if (config_.layer_types[layer] == LayerKind::conv) {
            std::vector<float> bcx(N * 3 * H);
            checkpoint::project_source(
                *resources_,
                Workload::lfm3_vision,
                compute_.get(),
                *shard_,
                lp + "conv.in_proj.weight",
                {},
                normed,
                H,
                3 * H,
                bcx,
                cancel
            );
            // Scatter from merge-block order to raster [H, grid_h, grid_w].
            std::vector<float> raster(H * patches.grid_h * patches.grid_w, 0.0f);
            std::size_t idx = 0;
            for (std::size_t bh = 0; bh < patches.grid_h / merge; ++bh) {
                for (std::size_t bw = 0; bw < patches.grid_w / merge; ++bw) {
                    for (std::size_t mh = 0; mh < merge; ++mh) {
                        for (std::size_t mw = 0; mw < merge; ++mw) {
                            const std::size_t gy = bh * merge + mh;
                            const std::size_t gx = bw * merge + mw;
                            const float* row = bcx.data() + idx * 3 * H;
                            for (std::size_t c = 0; c < H; ++c)
                                raster[(c * patches.grid_h + gy) * patches.grid_w + gx] =
                                    row[c] * row[2 * H + c];
                            ++idx;
                        }
                    }
                }
            }
            const std::size_t K = config_.conv_kernel_size;
            const std::ptrdiff_t pad = static_cast<std::ptrdiff_t>(K / 2);
            auto conv_w = checkpoint::float_slice(
                *resources_,
                Workload::lfm3_vision,
                *shard_,
                lp + "conv.spatial_conv.weight",
                0,
                H * K * K,
                cancel
            );
            std::vector<float> conv_raster(H * patches.grid_h * patches.grid_w, 0.0f);
            for (std::size_t c = 0; c < H; ++c) {
                const float* wc = conv_w.data() + c * K * K;
                const float* src_c = raster.data() + c * patches.grid_h * patches.grid_w;
                float* dst_c = conv_raster.data() + c * patches.grid_h * patches.grid_w;
                for (std::size_t gy = 0; gy < patches.grid_h; ++gy) {
                    for (std::size_t gx = 0; gx < patches.grid_w; ++gx) {
                        double acc = 0.0;
                        for (std::size_t ky = 0; ky < K; ++ky) {
                            const std::ptrdiff_t sy =
                                static_cast<std::ptrdiff_t>(gy) + static_cast<std::ptrdiff_t>(ky)
                                - pad;
                            if (sy < 0 || sy >= static_cast<std::ptrdiff_t>(patches.grid_h))
                                continue;
                            for (std::size_t kx = 0; kx < K; ++kx) {
                                const std::ptrdiff_t sx =
                                    static_cast<std::ptrdiff_t>(gx)
                                    + static_cast<std::ptrdiff_t>(kx) - pad;
                                if (sx < 0
                                    || sx >= static_cast<std::ptrdiff_t>(patches.grid_w))
                                    continue;
                                acc += static_cast<double>(
                                           src_c[static_cast<std::size_t>(sy) * patches.grid_w
                                                 + static_cast<std::size_t>(sx)]
                                       )
                                       * static_cast<double>(wc[ky * K + kx]);
                            }
                        }
                        dst_c[gy * patches.grid_w + gx] = static_cast<float>(acc);
                    }
                }
            }
            // Gather back to merge-block order and multiply by C gate.
            std::vector<float> gated_out(N * H);
            idx = 0;
            for (std::size_t bh = 0; bh < patches.grid_h / merge; ++bh) {
                for (std::size_t bw = 0; bw < patches.grid_w / merge; ++bw) {
                    for (std::size_t mh = 0; mh < merge; ++mh) {
                        for (std::size_t mw = 0; mw < merge; ++mw) {
                            const std::size_t gy = bh * merge + mh;
                            const std::size_t gx = bw * merge + mw;
                            const float* c_gate = bcx.data() + idx * 3 * H + H;
                            float* dst = gated_out.data() + idx * H;
                            for (std::size_t c = 0; c < H; ++c)
                                dst[c] = c_gate[c]
                                         * conv_raster[(c * patches.grid_h + gy) * patches.grid_w
                                                       + gx];
                            ++idx;
                        }
                    }
                }
            }
            checkpoint::project_source(
                *resources_,
                Workload::lfm3_vision,
                compute_.get(),
                *shard_,
                lp + "conv.out_proj.weight",
                {},
                gated_out,
                H,
                H,
                op_out,
                cancel
            );
        } else {
            const std::string ap = lp + "self_attn.";
            std::vector<float> q(N * H), k(N * kv_dim), v(N * kv_dim);
            checkpoint::project_source(
                *resources_, Workload::lfm3_vision, compute_.get(), *shard_,
                ap + "q_proj.weight", {}, normed, H, H, q, cancel
            );
            checkpoint::project_source(
                *resources_, Workload::lfm3_vision, compute_.get(), *shard_,
                ap + "k_proj.weight", {}, normed, H, kv_dim, k, cancel
            );
            checkpoint::project_source(
                *resources_, Workload::lfm3_vision, compute_.get(), *shard_,
                ap + "v_proj.weight", {}, normed, H, kv_dim, v, cancel
            );
            auto qn = checkpoint::float_slice(
                *resources_, Workload::lfm3_vision, *shard_, ap + "q_layernorm.weight",
                0, head_dim, cancel
            );
            auto kn = checkpoint::float_slice(
                *resources_, Workload::lfm3_vision, *shard_, ap + "k_layernorm.weight",
                0, head_dim, cancel
            );
            rms_norm_lfm3(q, qn, N * config_.num_attention_heads, head_dim, config_.norm_eps, q);
            rms_norm_lfm3(k, kn, N * config_.num_key_value_heads, head_dim, config_.norm_eps, k);
            apply_axial_rope(q, config_.num_attention_heads);
            apply_axial_rope(k, config_.num_key_value_heads);

            std::vector<float> attn_out(N * H);
            bool gpu_done = false;
            if (compute_) {
                gpu_done = compute_->attend(
                    q, k, v, N, N, config_.num_attention_heads,
                    config_.num_key_value_heads, head_dim, 0, 0, 0, attn_out, cancel
                );
            }
            if (!gpu_done) {
                const std::size_t group =
                    config_.num_attention_heads / config_.num_key_value_heads;
                const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                std::vector<float> scores(N);
                for (std::size_t qi = 0; qi < N; ++qi) {
                    if ((qi & 15) == 0)
                        check_cancel(cancel);
                    for (std::size_t h = 0; h < config_.num_attention_heads; ++h) {
                        const std::size_t kh = h / group;
                        const float* qp = q.data() + (qi * config_.num_attention_heads + h) * head_dim;
                        float mx = -std::numeric_limits<float>::infinity();
                        for (std::size_t ki = 0; ki < N; ++ki) {
                            const float* kp =
                                k.data() + (ki * config_.num_key_value_heads + kh) * head_dim;
                            double dot = 0.0;
                            for (std::size_t d = 0; d < head_dim; ++d)
                                dot += static_cast<double>(qp[d]) * kp[d];
                            scores[ki] = static_cast<float>(dot) * scale;
                            mx = std::max(mx, scores[ki]);
                        }
                        double sum = 0.0;
                        for (std::size_t ki = 0; ki < N; ++ki) {
                            scores[ki] = static_cast<float>(std::exp(scores[ki] - mx));
                            sum += scores[ki];
                        }
                        const float inv = static_cast<float>(1.0 / sum);
                        float* dst =
                            attn_out.data() + (qi * config_.num_attention_heads + h) * head_dim;
                        std::fill(dst, dst + head_dim, 0.0f);
                        for (std::size_t ki = 0; ki < N; ++ki) {
                            const float w = scores[ki] * inv;
                            const float* vp =
                                v.data() + (ki * config_.num_key_value_heads + kh) * head_dim;
                            for (std::size_t d = 0; d < head_dim; ++d)
                                dst[d] += w * vp[d];
                        }
                    }
                }
            }
            checkpoint::project_source(
                *resources_, Workload::lfm3_vision, compute_.get(), *shard_,
                ap + "out_proj.weight", {}, attn_out, H, H, op_out, cancel
            );
        }

        for (std::size_t i = 0; i < N * H; ++i)
            hidden[i] += op_out[i];

        auto ffn_nw = checkpoint::float_slice(
            *resources_, Workload::lfm3_vision, *shard_, lp + "ffn_norm.weight", 0, H, cancel
        );
        rms_norm_lfm3(hidden, ffn_nw, N, H, config_.norm_eps, normed);
        checkpoint::project_source(
            *resources_, Workload::lfm3_vision, compute_.get(), *shard_,
            lp + "feed_forward.w1.weight", {}, normed, H, config_.intermediate_size, ffn_gate,
            cancel
        );
        checkpoint::project_source(
            *resources_, Workload::lfm3_vision, compute_.get(), *shard_,
            lp + "feed_forward.w3.weight", {}, normed, H, config_.intermediate_size, ffn_up,
            cancel
        );
        for (std::size_t i = 0; i < N * config_.intermediate_size; ++i)
            ffn_gate[i] = static_cast<float>(silu_f64(ffn_gate[i])) * ffn_up[i];
        checkpoint::project_source(
            *resources_, Workload::lfm3_vision, compute_.get(), *shard_,
            lp + "feed_forward.w2.weight", {}, ffn_gate, config_.intermediate_size, H, op_out,
            cancel
        );
        for (std::size_t i = 0; i < N * H; ++i)
            hidden[i] += op_out[i];
    }

    // 3. Multimodal projector: LayerNorm -> reshape merge^2 -> linear_1 -> GELU -> linear_2.
    {
        auto ln_w = checkpoint::float_slice(
            *resources_, Workload::lfm3_vision, *shard_,
            "model.multi_modal_projector.layer_norm.weight", 0, H, cancel
        );
        auto ln_b = checkpoint::float_slice(
            *resources_, Workload::lfm3_vision, *shard_,
            "model.multi_modal_projector.layer_norm.bias", 0, H, cancel
        );
        layer_norm_1d(hidden, ln_w, ln_b, N, H, config_.projector_layer_norm_eps, hidden);
    }
    const std::size_t group_sz = config_.downsample_factor * config_.downsample_factor;
    const std::size_t merged_tokens = N / group_sz;
    const std::size_t proj_in = H * group_sz;
    std::vector<float> proj_mid(merged_tokens * config_.projector_hidden_size);
    {
        std::vector<float> b1;
        if (config_.projector_bias) {
            auto b = checkpoint::float_slice(
                *resources_, Workload::lfm3_vision, *shard_,
                "model.multi_modal_projector.linear_1.bias", 0,
                config_.projector_hidden_size, cancel
            );
            b1.assign(b.begin(), b.end());
        }
        checkpoint::project_source(
            *resources_, Workload::lfm3_vision, compute_.get(), *shard_,
            "model.multi_modal_projector.linear_1.weight", b1, hidden, proj_in,
            config_.projector_hidden_size, proj_mid, cancel
        );
    }
    for (float& v : proj_mid)
        v = static_cast<float>(gelu_f64(v));

    std::vector<float> projected(merged_tokens * config_.text_hidden_size);
    {
        std::vector<float> b2;
        if (config_.projector_bias) {
            auto b = checkpoint::float_slice(
                *resources_, Workload::lfm3_vision, *shard_,
                "model.multi_modal_projector.linear_2.bias", 0,
                config_.text_hidden_size, cancel
            );
            b2.assign(b.begin(), b.end());
        }
        checkpoint::project_source(
            *resources_, Workload::lfm3_vision, compute_.get(), *shard_,
            "model.multi_modal_projector.linear_2.weight", b2, proj_mid,
            config_.projector_hidden_size, config_.text_hidden_size, projected, cancel
        );
    }
    return projected;
}

}  // namespace kadan::lfm3
