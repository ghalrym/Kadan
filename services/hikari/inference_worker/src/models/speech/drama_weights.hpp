#pragma once

#include <map>
#include <set>
#include <vector>

#include "kadan/checkpoint.hpp"
#include "kadan/drama_model.hpp"

namespace kadan::drama {
using Vector = std::vector<float>;

void require(bool condition, const char* message);
void check_cancel(const std::atomic_bool&);

// Strict original Drama-3 safetensors schema; no whole-model allocation. The
// shared checkpoint parser bounds headers through its MemoryBudget.
class Weights {
public:
    Weights(const std::filesystem::path&, const ModelInfo&, const std::atomic_bool&);
    void validate(const std::atomic_bool&);
    Bytes execution_bytes() const;
    void check_unchanged() const;
    Vector read(const std::string&, const std::atomic_bool&) const;
    Vector row(const std::string&, std::size_t, const std::atomic_bool&) const;
    Vector project(
        DenseCompute&,
        const std::string&,
        std::span<const float>,
        const std::atomic_bool&,
        std::size_t first_row = 0,
        std::size_t rows = 0
    );
    void release(DenseCompute&);

private:
    struct Tensor {
        checkpoint::Shard* shard;
        checkpoint::TensorInfo info;
    };

    const Tensor& tensor(const std::string&) const;
    void
    shape(const std::string&, std::initializer_list<std::uint64_t>, bool fp8 = false);
    void decode(
        const std::string&, std::size_t, std::span<float>, const std::atomic_bool&
    ) const;
    std::shared_ptr<checkpoint::MemoryBudget> metadata_;
    std::vector<std::unique_ptr<checkpoint::Shard>> shards_;
    std::map<std::string, Tensor> tensors_;
    std::set<std::string> bound_;
    std::set<WeightIdentity> used_;
    ModelInfo info_;
};
}  // namespace kadan::drama
