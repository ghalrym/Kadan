#include <array>
#include <stdexcept>

// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright 2025-2026 SGLang-Omni Team
// Copyright (c) 2023-present, Descript
// Canonical decoder tensor schema adapted from Fish's SGLang-Omni codec:
// https://github.com/sgl-project/sglang-omni/blob/be34dd84ed9e0971f72fe5e3d9ba8245cdad780a/sglang_omni/models/fishaudio_s2_pro/fish_speech/configs/modded_dac_vq.yaml
// https://github.com/sgl-project/sglang-omni/blob/be34dd84ed9e0971f72fe5e3d9ba8245cdad780a/sglang_omni/models/fishaudio_s2_pro/fish_speech/models/dac/modded_dac.py
// See drama_codec.cpp for the quantizer's Descript DAC source attribution.
// Changes: expose only required decoder tensors after weight-norm folding.
#include "kadan/drama_codec.hpp"

namespace kadan::drama {
std::vector<CodecTensorSpec> codec_tensor_specs() {
    std::vector<CodecTensorSpec> s;
    auto add = [&](std::string p, std::initializer_list<std::uint64_t> shape) {
        std::size_t count = 1;
        for (auto n : shape)
            count *= n;
        s.push_back({std::move(p), shape, count});
    };
    auto linear = [&](std::string p, std::size_t in, std::size_t out, bool bias) {
        add(p + ".weight", {out, in});
        if (bias)
            add(p + ".bias", {out});
    };
    auto conv = [&](std::string p,
                    std::size_t in,
                    std::size_t out,
                    std::size_t k,
                    bool transpose = false,
                    bool depth = false) {
        add(p + ".conv.weight",
            {transpose ? in : out,
             depth       ? 1
             : transpose ? out
                         : in,
             k});
        add(p + ".conv.bias", {out});
    };
    auto snake = [&](std::string p, std::size_t n) {
        add(p + ".alpha", {1, n, 1});
    };
    for (std::size_t i = 0; i < 10; ++i) {
        const auto p = i == 0
                           ? std::string("quantizer.semantic_quantizer.quantizers.0")
                           : "quantizer.quantizer.quantizers." + std::to_string(i - 1);
        add(p + ".codebook.weight", {i == 0 ? 4096U : 1024U, 8});
        add(p + ".out_proj.weight", {1024, 8, 1});
        add(p + ".out_proj.bias", {1024});
    }
    add("quantizer.post_module.norm.weight", {1024});
    for (int i = 0; i < 8; ++i) {
        const auto p = "quantizer.post_module.layers." + std::to_string(i);
        for (auto n :
             {"attention_norm.weight",
              "ffn_norm.weight",
              "attention_layer_scale.gamma",
              "ffn_layer_scale.gamma"})
            add(p + "." + n, {1024});
        linear(p + ".attention.wqkv", 1024, 3072, false);
        linear(p + ".attention.wo", 1024, 1024, false);
        linear(p + ".feed_forward.w1", 1024, 3072, false);
        linear(p + ".feed_forward.w3", 1024, 3072, false);
        linear(p + ".feed_forward.w2", 3072, 1024, false);
    }
    for (int i = 0; i < 2; ++i) {
        const auto p = "quantizer.upsample." + std::to_string(i);
        conv(p + ".0", 1024, 1024, 2, true);
        conv(p + ".1.dwconv", 1024, 1024, 7, false, true);
        add(p + ".1.norm.weight", {1024});
        add(p + ".1.norm.bias", {1024});
        add(p + ".1.gamma", {1024});
        linear(p + ".1.pwconv1", 1024, 4096, true);
        linear(p + ".1.pwconv2", 4096, 1024, true);
    }
    conv("decoder.model.0", 1024, 1536, 7);
    constexpr std::array<std::size_t, 4> rates{8, 8, 4, 2};
    for (std::size_t i = 0; i < rates.size(); ++i) {
        const auto in = 1536U >> i, out = in / 2;
        const auto p = "decoder.model." + std::to_string(i + 1) + ".block";
        snake(p + ".0", in);
        conv(p + ".1", in, out, 2 * rates[i], true);
        for (int j = 2; j < 5; ++j) {
            const auto q = p + "." + std::to_string(j) + ".block";
            snake(q + ".0", out);
            conv(q + ".1", out, out, 7);
            snake(q + ".2", out);
            conv(q + ".3", out, out, 1);
        }
    }
    snake("decoder.model.5", 96);
    conv("decoder.model.6", 96, 1, 7);
    return s;
}

std::size_t codec_window_samples(std::span<const std::uint32_t> codes) {
    if (codes.empty() || codes.size() % Codec::codebooks != 0
        || codes.size() / Codec::codebooks > Codec::max_frames)
        throw std::invalid_argument("drama_codec_code_shape");
    for (auto code : codes)
        if (code >= 4096U)
            throw std::invalid_argument("drama_codec_code_range");
    return codes.size() / Codec::codebooks * Codec::samples_per_frame;
}

}  // namespace kadan::drama
