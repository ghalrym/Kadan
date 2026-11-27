#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace kadan {
enum class InklingVariant : std::uint8_t {
    base = 0,
    abliterated = 1
};

inline std::optional<InklingVariant> parse_inkling_variant(std::string_view value) {
    if (value == "base")
        return InklingVariant::base;
    if (value == "abliterated")
        return InklingVariant::abliterated;
    return std::nullopt;
}

inline std::string_view inkling_variant_name(InklingVariant variant) {
    return variant == InklingVariant::abliterated ? "abliterated" : "base";
}

inline std::optional<InklingVariant> inkling_model_variant(std::string_view model_id) {
    if (model_id == "inkling-2-mini-iq4-nl" || model_id == "inkling-2-mini-q8-0")
        return InklingVariant::base;
    if (model_id == "inkling-2-mini-abliterated-iq4-nl"
        || model_id == "inkling-2-mini-abliterated-q8-0")
        return InklingVariant::abliterated;
    return std::nullopt;
}

inline std::optional<std::string_view> inkling_model_quant(std::string_view model_id) {
    if (model_id == "inkling-2-mini-iq4-nl"
        || model_id == "inkling-2-mini-abliterated-iq4-nl")
        return "iq4-nl";
    if (model_id == "inkling-2-mini-q8-0"
        || model_id == "inkling-2-mini-abliterated-q8-0")
        return "q8-0";
    return std::nullopt;
}

inline std::string resolve_inkling_variant_model(
    std::string_view selected_model_id, InklingVariant variant
) {
    const auto quant = inkling_model_quant(selected_model_id);
    const bool q8 = quant && *quant == "q8-0";
    if (variant == InklingVariant::abliterated)
        return q8 ? "inkling-2-mini-abliterated-q8-0"
                  : "inkling-2-mini-abliterated-iq4-nl";
    return q8 ? "inkling-2-mini-q8-0" : "inkling-2-mini-iq4-nl";
}

inline bool
inkling_same_precision_family(std::string_view model_a, std::string_view model_b) {
    const auto qa = inkling_model_quant(model_a);
    const auto qb = inkling_model_quant(model_b);
    return qa.has_value() && qb.has_value() && *qa == *qb;
}

inline bool
inkling_hot_swap_compatible(std::string_view model_a, std::string_view model_b) {
    const auto va = inkling_model_variant(model_a);
    const auto vb = inkling_model_variant(model_b);
    return va.has_value() && vb.has_value() && *va != *vb
           && inkling_same_precision_family(model_a, model_b);
}

inline std::uint32_t inkling_variant_numeric_id(std::string_view variant_name) {
    return variant_name == "abliterated" ? 1u : 0u;
}

// Physical layer ranges where Inkling 2 Mini abliteration modifies weights
// (34 of 704 tensors; remaining 670 tensors are byte-for-byte identical).
inline constexpr std::size_t kInklingVariantAttnOutputStart = 8;
inline constexpr std::size_t kInklingVariantFfnDownStart = 6;
inline constexpr std::size_t kInklingVariantPhysLayers = 24;

inline bool is_inkling_variant_tensor(std::string_view name) {
    if (!name.starts_with("blk."))
        return false;
    const auto dot = name.find('.', 4);
    if (dot == std::string_view::npos || dot == 4)
        return false;
    std::size_t layer = 0;
    for (std::size_t i = 4; i < dot; ++i) {
        const char ch = name[i];
        if (ch < '0' || ch > '9')
            return false;
        layer = layer * 10 + std::size_t(ch - '0');
        if (layer >= kInklingVariantPhysLayers)
            return false;
    }
    const auto suffix = name.substr(dot + 1);
    if (suffix == "attn_output.weight")
        return layer >= kInklingVariantAttnOutputStart;
    if (suffix == "ffn_down_exps.weight")
        return layer >= kInklingVariantFfnDownStart;
    return false;
}
}  // namespace kadan
