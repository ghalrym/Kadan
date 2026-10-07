#pragma once
#include <cstddef>
#include <span>

namespace kadan::math {
enum class NormScale { direct, one_plus };
enum class Gate { sigmoid, silu };
struct Norm { std::size_t rows, width; float epsilon; NormScale scale; };
struct Rotary { std::size_t heads, head_dim, rotary_dim, context; };
// Original FP32 math references; no model loading. Norm reduces each row across
// width with shared per-channel weights. A positive finite epsilon is mandatory.
// Output may exactly alias input (or a gate); other overlaps are rejected.
// Finite inputs/outputs required; discard output after an exception.
void rms_norm(Norm shape,std::span<const float> input,std::span<const float> weight,std::span<float> output);
void gate(Gate kind,std::span<const float> input,std::span<const float> modulation,std::span<float> output);
// Prepare once, then retain/upload these FP32 frequencies. Length = rotary_dim/2.
void inverse_frequencies(double theta,std::span<float> output);
// Text-only common scalar position across heads and T/H/W streams. Split-half
// pairs within the rotary prefix; suffix passes through unchanged. No YaRN/MRoPE
// spatial positions or dtype conversions. Angle is explicitly rounded to FP32.
void rope(Rotary shape,std::size_t position,std::span<const float> frequencies,
          std::span<const float> input,std::span<float> output);
} // namespace kadan::math
