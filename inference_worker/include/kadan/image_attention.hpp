#pragma once

#include <stdexcept>
#include <string_view>

#include "kadan/resources.hpp"

namespace kadan::image {
enum class AttentionPrecision {
    pedantic_fp32,
    experimental_pv_tf32
};

inline std::string_view attention_precision_name(AttentionPrecision precision) {
    switch (precision) {
    case AttentionPrecision::pedantic_fp32:
        return "pedantic_fp32";
    case AttentionPrecision::experimental_pv_tf32:
        return "experimental_pv_tf32";
    }
    throw std::invalid_argument("image_attention_precision");
}

inline void
validate_attention_precision(AttentionPrecision precision, Workload workload) {
    attention_precision_name(precision);
    if (precision == AttentionPrecision::experimental_pv_tf32
        && workload != Workload::image)
        throw std::invalid_argument("image_attention_precision_workload");
}
}  // namespace kadan::image
