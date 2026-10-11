#pragma once
#include "kadan/auto_placement.hpp"
#include "kadan/cuda_model.hpp"
namespace kadan::cuda {
// Existing Qwen kernels, complete decoder layers, explicit bounded transfers.
// Owns its process's CUDA contexts exclusively, like the resident Model.
class AutoModel {
public:
  AutoModel(const char *root, ModelOptions,
            const std::map<int, std::size_t> &budgets,
            std::shared_ptr<Resources>,
            const std::atomic_bool *cancel = nullptr);
  ~AutoModel();
  void set_cancel(const std::atomic_bool &);
  void begin_request();
  void end_request();
  void park();
  void close();
  void reset();
  stack::Selection step(unsigned, bool);
  std::size_t tokens() const;
  serving::WeightBacking::Stats cache_stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
inline model::AutoPlacement
automatic_layout(const model::Layout &layout,
                 const std::map<int, std::size_t> &budgets,
                 std::size_t headroom) {
  std::vector<model::LayerBytes> layers;
  for (const auto &l : layout.layers()) {
    auto w = l.plan.moe_offset + l.plan.moe.scratch_offset;
    layers.push_back({w, l.plan.device_bytes - w});
  }
  return model::auto_place(layers, layout.embedded() - layout.embedding(),
                           layout.device_bytes() - layout.embedded(),
                           layout.manifest().architecture().hidden * 4, budgets,
                           headroom);
}
} // namespace kadan::cuda
