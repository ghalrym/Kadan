#pragma once
// Host products of packed GGUF rows using ggml-cpu's quantized dot kernels, for
// routed experts that live in host RAM. Each input row is quantized once to the
// format's dot type (as llama.cpp does on the CPU), so results differ from the
// device's F32 products by that activation rounding.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace kadan::text {
class CpuMatvec {
public:
    // False when ggml-cpu has no dot kernel for this GGML tensor type, or rows
    // of `in` values are not whole blocks of it and of its dot type.
    static bool supported(std::uint32_t type, std::size_t in = 0);

    // Quantizes rows of `in` values once for every later run().
    CpuMatvec(std::uint32_t type, std::span<const float> input, std::size_t in);

    std::size_t rows() const {
        return rows_;
    }

    // output[r * out + c] for c in [first, end); `weights` holds out packed rows.
    void
    run(const std::byte* weights,
        std::size_t row_bytes,
        std::size_t first,
        std::size_t end,
        std::span<float> output,
        std::size_t out) const;

private:
    std::uint32_t type_;
    std::size_t in_, rows_, row_bytes_;
    std::vector<std::byte> input_;
};
}  // namespace kadan::text
