#pragma once

#include <stdexcept>
#include <string_view>

namespace kadan::image {
enum class AttentionPrecision {
    pedantic_fp32
};

inline std::string_view attention_precision_name(AttentionPrecision precision) {
    switch (precision) {
    case AttentionPrecision::pedantic_fp32:
        return "pedantic_fp32";
    }
    throw std::invalid_argument("image_attention_precision");
}

}  // namespace kadan::image
