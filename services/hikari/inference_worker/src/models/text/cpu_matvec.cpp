#include "cpu_matvec.hpp"

#include <cstring>
#include <ggml-cpu.h>
#include <ggml.h>
#include <mutex>
#include <stdexcept>

namespace kadan::text {
namespace {
const ggml_type_traits_cpu& traits(std::uint32_t type) {
    static std::once_flag initialized;
    std::call_once(initialized, [] { ggml_cpu_init(); });
    return *ggml_get_type_traits_cpu(ggml_type(type));
}
}  // namespace

bool CpuMatvec::supported(std::uint32_t type, std::size_t in) {
    if (type >= GGML_TYPE_COUNT || !ggml_get_type_traits(ggml_type(type))->blck_size)
        return false;
    const auto& t = traits(type);
    return t.vec_dot
           && (t.vec_dot_type == GGML_TYPE_F32 || traits(t.vec_dot_type).from_float)
           && in % ggml_blck_size(ggml_type(type)) == 0
           && in % ggml_blck_size(t.vec_dot_type) == 0;
}

CpuMatvec::CpuMatvec(std::uint32_t type, std::span<const float> input, std::size_t in)
    : type_(type), in_(in), rows_(in ? input.size() / in : 0) {
    if (!in || !supported(type, in) || input.size() % in)
        throw std::invalid_argument("cpu_matvec_shape");
    const auto dot = traits(type).vec_dot_type;
    row_bytes_ = ggml_row_size(dot, std::int64_t(in));
    input_.resize(rows_ * row_bytes_);
    for (std::size_t r = 0; r < rows_; ++r) {
        auto* to = input_.data() + r * row_bytes_;
        if (dot == GGML_TYPE_F32)
            std::memcpy(to, input.data() + r * in, row_bytes_);
        else
            traits(dot).from_float(input.data() + r * in, to, std::int64_t(in));
    }
}

void CpuMatvec::run(
    const std::byte* weights,
    std::size_t row_bytes,
    std::size_t first,
    std::size_t end,
    std::span<float> output,
    std::size_t out
) const {
    if (end > out || first > end || output.size() != rows_ * out
        || row_bytes != ggml_row_size(ggml_type(type_), std::int64_t(in_)))
        throw std::invalid_argument("cpu_matvec_range");
    const auto dot = traits(type_).vec_dot;
    for (std::size_t c = first; c < end; ++c)
        for (std::size_t r = 0; r < rows_; ++r)
            dot(int(in_),
                output.data() + r * out + c,
                0,
                weights + c * row_bytes,
                0,
                input_.data() + r * row_bytes_,
                0,
                1);
}
}  // namespace kadan::text
