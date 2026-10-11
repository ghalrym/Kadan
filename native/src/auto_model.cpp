#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "Model requires legacy default stream"
#endif
#include "kadan/auto_model.hpp"
#include "decoder_producer.hpp"
#include "stack_kernel.cuh"
#include <list>
#include <optional>
#include <thread>
namespace kadan::cuda {
namespace {
void require(bool v, const char *e) {
  if (!v)
    throw std::invalid_argument(e);
}
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace
struct AutoModel::Impl final : model::Sink {
  ModelOptions options;
  std::shared_ptr<Resources> resources;
  std::shared_ptr<checkpoint::MemoryBudget> metadata, staging;
  std::optional<checkpoint::ModelManifest> manifest;
  std::optional<model::Layout> layout;
  std::optional<StateCursor> cursor;
  model::AutoPlacement plan;
  struct Allocation {
    void *weights = nullptr;
    void *state = nullptr;
    bool touched = false;
  };
  std::map<int, Allocation> gpu;
  std::pmr::list<detail::DecoderProducer> producers;
  std::array<detail::DecoderProducer *, 256> layers{};
  std::pmr::vector<float> transfer;
  std::shared_ptr<serving::WeightBacking> backing;
  std::thread::id thread = std::this_thread::get_id();
  Handle host = 0, weights = 0, state = 0;
  bool host_loaded = false, weights_loaded = false, state_loaded = false,
       poisoned = false, ended = false;
  float head_global = 0;
  const std::atomic_bool *cancel = nullptr;
  void check_cancel() const {
    if (cancel && cancel->load())
      throw std::runtime_error("model_cancelled");
  }
  Impl(ModelOptions o, std::shared_ptr<Resources> r)
      : options(o), resources(std::move(r)),
        metadata(std::make_shared<checkpoint::MemoryBudget>(o.metadata_bytes)),
        staging(std::make_shared<checkpoint::MemoryBudget>(o.staging_bytes)),
        producers(metadata.get()), transfer(metadata.get()) {}
  void current(int d) {
    require(thread == std::this_thread::get_id(), "model_thread_changed");
    check(cudaSetDevice(d));
  }
  Footprint footprint() {
    auto v = resources->snapshot().capacity;
    std::fill(v.begin(), v.end(), 0);
    return v;
  }
  std::uint8_t *global(std::size_t offset) {
    auto &a = gpu.at(plan.primary);
    return offset < layout->embedded()
               ? static_cast<std::uint8_t *>(a.weights) + offset -
                     layout->embedding()
               : static_cast<std::uint8_t *>(a.state) + offset -
                     layout->embedded();
  }
  template <class T> T *pointer(std::size_t offset) {
    return reinterpret_cast<T *>(global(offset));
  }
  void write(std::size_t offset, std::span<const std::uint8_t> data) override {
    check_cancel();
    if (offset >= layout->embedding()) {
      require(offset < layout->embedded() &&
                  data.size() <= layout->embedded() - offset,
              "global_upload_bounds");
      current(plan.primary);
      check(cudaMemcpy(global(offset), data.data(), data.size(),
                       cudaMemcpyHostToDevice));
      return;
    }
    auto ls = layout->layers();
    auto it =
        std::upper_bound(ls.begin(), ls.end(), offset,
                         [](auto x, const auto &l) { return x < l.offset; });
    require(it != ls.begin(), "layer_upload_offset");
    --it;
    auto i = std::size_t(it - ls.begin());
    auto split = it->plan.moe_offset + it->plan.moe.scratch_offset;
    auto local = offset - it->offset;
    require(local < split && data.size() <= split - local,
            "layer_upload_bounds");
    current(plan.devices[i]);
    check(cudaMemcpy(layers[i]->bytes(local), data.data(), data.size(),
                     cudaMemcpyHostToDevice));
  }
  void multiplier(const model::Binding &b, float value) override {
    if (b.layer < 0)
      head_global = value;
    else
      layers.at(std::size_t(b.layer))
          ->experts.at(std::size_t(b.expert))
          .at(std::size_t(b.part))
          .global = value;
  }
  void initialize(const char *root, const std::map<int, std::size_t> &budgets) {
    check_cancel();
    auto h = footprint();
    h[0] = Model::host_bytes(options);
    host = resources->reserve(Workload::llm, h);
    manifest.emplace(root, metadata);
    check_cancel();
    auto generation =
        model::read_generation(root, manifest->architecture().vocab, metadata);
    layout.emplace(*manifest, options.capacity, generation, metadata);
    require(options.staging_bytes >= layout->minimum_staging_bytes(),
            "model_staging_row");
    plan = automatic_layout(*layout, budgets, options.device_headroom);
    cursor.emplace(layout->layers().size(), options.capacity);
    transfer.resize(manifest->architecture().hidden);
    auto used = resources->snapshot().used;
    for (auto [d, a] : plan.allocations) {
      (void)a;
      require(used.at(d + 1) == 0, "split_requires_exclusive_context");
      h.at(d + 1) = options.device_headroom;
      gpu.emplace(d, Allocation{});
    }
    resources->resize_loading(host, h);
    for (auto &[d, a] : gpu) {
      a.touched = true;
      current(d);
      int major = 0, minor = 0;
      check(
          cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d));
      check(
          cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, d));
      require(major == 8 && minor == 6, "requires_sm86");
    }
    for (std::size_t i = 0; i < layout->layers().size(); ++i) {
      const auto &l = layout->layers()[i];
      producers.emplace_back(l.config, l.plan, *cursor);
      layers[i] = &producers.back();
    }
    backing = std::make_shared<serving::WeightBacking>(
        resources, options.weight_ram_bytes, options.weight_cold_bytes,
        options.staging_bytes, serving::WeightBacking::model_entry_limit, true);
    manifest->backing(backing);
    begin_request();
    resources->loaded(host);
    host_loaded = true;
  }
  void bind(bool preserve) {
    for (std::size_t i = 0; i < layout->layers().size(); ++i) {
      auto *p = layers[i];
      auto &a = gpu.at(plan.devices[i]);
      p->storage =
          static_cast<std::uint8_t *>(a.weights) + plan.weight_offsets[i];
      p->request_storage =
          static_cast<std::uint8_t *>(a.state) + plan.state_offsets[i];
      p->split = p->p.moe_offset + p->p.moe.scratch_offset;
      p->bind(preserve);
    }
  }
  void begin_request() {
    check_cancel();
    require(!poisoned && !state, "model_request_state");
    const bool reload = !weights;
    require(backing->room_for_reservations(reload ? 2 : 1),
            "model_resident_slots");
    auto w = footprint(), s = footprint();
    for (auto [d, a] : plan.allocations) {
      w.at(d + 1) = a.weights;
      s.at(d + 1) = a.state;
    }
    try {
      if (reload)
        weights = resources->reserve(Workload::llm, w);
      state = resources->reserve(Workload::llm, s);
      for (auto [d, a] : plan.allocations) {
        current(d);
        std::size_t free = 0, total = 0;
        check(cudaMemGetInfo(&free, &total));
        require(free >= model::checked_add(a.state, reload ? a.weights : 0),
                "model_physical_headroom");
        auto &allocation = gpu.at(d);
        if (reload)
          check(cudaMalloc(&allocation.weights, a.weights));
        check(cudaMalloc(&allocation.state, a.state));
      }
      bind(!reload);
      if (reload) {
        model::load_range(*layout, staging, *this, layout->embedding(),
                          layout->embedded());
        if (!plan.streaming)
          for (const auto &l : layout->layers())
            model::load_range(*layout, staging, *this, l.offset,
                              l.offset + l.plan.device_bytes);
      }
      reset_state();
      if (reload) {
        resources->loaded(weights);
        weights_loaded = true;
      }
      resources->loaded(state);
      state_loaded = true;
    } catch (...) {
      poisoned = true;
      throw;
    }
  }
  void reset_state() {
    for (std::size_t i = 0; i < layout->layers().size(); ++i) {
      current(plan.devices[i]);
      layers[i]->zero();
    }
    current(plan.primary);
    check(cudaMemsetAsync(global(layout->embedded()), 0,
                          layout->device_bytes() - layout->embedded(),
                          cudaStreamLegacy));
    for (auto [d, a] : gpu) {
      (void)a;
      current(d);
      check(cudaStreamSynchronize(cudaStreamLegacy));
    }
    cursor->reset();
    ended = false;
    current(plan.primary);
  }
  void release(Handle &h, bool &loaded) {
    if (!h)
      return;
    if (loaded)
      resources->begin_eviction(h);
    resources->released(h);
    h = 0;
    loaded = false;
  }
  void end_request() {
    try {
      for (auto &[d, a] : gpu) {
        if (!a.touched)
          continue;
        current(d);
        check(cudaStreamSynchronize(cudaStreamLegacy));
        if (a.state) {
          check(cudaFree(a.state));
          a.state = nullptr;
        }
      }
      release(state, state_loaded);
      if (cursor)
        cursor->invalidate();
      if (gpu.contains(plan.primary) && gpu.at(plan.primary).touched)
        current(plan.primary);
    } catch (...) {
      poisoned = true;
      throw;
    }
  }
  void park() {
    end_request();
    try {
      for (auto &[d, a] : gpu) {
        if (!a.touched)
          continue;
        current(d);
        if (a.weights) {
          check(cudaFree(a.weights));
          a.weights = nullptr;
        }
      }
      release(weights, weights_loaded);
      if (gpu.contains(plan.primary) && gpu.at(plan.primary).touched)
        current(plan.primary);
    } catch (...) {
      poisoned = true;
      throw;
    }
  }
  void status() {
    check(cudaStreamSynchronize(cudaStreamLegacy));
    unsigned flags = 0;
    check(cudaMemcpy(&flags, global(layout->status()), 4,
                     cudaMemcpyDeviceToHost));
    if (flags)
      throw std::overflow_error("model_numeric_failure");
  }
  const float *move_input(const float *x, int from, int to) {
    if (from == to) {
      current(to);
      return x;
    }
    current(from);
    check(cudaStreamSynchronize(cudaStreamLegacy));
    check(cudaMemcpy(transfer.data(), x, transfer.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
    current(to);
    auto offset =
        to == plan.primary ? layout->device_bytes() - layout->embedded() : 0;
    auto *dest = static_cast<std::uint8_t *>(gpu.at(to).state) + offset;
    check(cudaMemcpy(dest, transfer.data(), transfer.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    return reinterpret_cast<const float *>(dest);
  }
  stack::Selection step(unsigned input, bool stop) {
    require(!poisoned && state && weights && !ended, "model_unavailable");
    auto token = cursor->begin();
    try {
      const auto &a = manifest->architecture();
      require(input < a.vocab, "model_input_id");
      current(plan.primary);
      check(cudaMemsetAsync(global(layout->status()), 0, 4, cudaStreamLegacy));
      check(detail::stack_embedding(a.hidden,
                                    pointer<std::uint16_t>(layout->embedding()),
                                    input, pointer<float>(layout->embedded())));
      status();
      const float *x = pointer<float>(layout->embedded());
      int previous = plan.primary;
      for (std::size_t i = 0; i < layout->layers().size(); ++i) {
        check_cancel();
        int d = plan.devices[i];
        x = move_input(x, previous, d);
        if (plan.streaming) {
          const auto &l = layout->layers()[i];
          model::load_range(*layout, staging, *this, l.offset,
                            l.offset + l.plan.device_bytes);
          current(d);
        }
        layers[i]->run(token, x, nullptr);
        x = layers[i]->work(3);
        previous = d;
        cursor->written(token, i);
      }
      x = move_input(x, previous, plan.primary);
      check(detail::decoder_norm(a.hidden, float(a.rms_epsilon),
                                 pointer<std::uint16_t>(layout->final_norm()),
                                 x, pointer<float>(layout->normalized()),
                                 pointer<unsigned>(layout->status())));
      status();
      check(kadan_launch_nvfp4_bf16(
          global(layout->head_weights()), global(layout->head_scales()),
          head_global, pointer<float>(layout->normalized()),
          pointer<float>(layout->logits()), pointer<unsigned>(layout->status()),
          a.vocab, a.hidden));
      status();
      check(detail::stack_select(a.vocab, pointer<float>(layout->logits()),
                                 pointer<unsigned>(layout->selected()),
                                 pointer<unsigned>(layout->status())));
      status();
      unsigned selected = 0;
      check(cudaMemcpy(&selected, global(layout->selected()), 4,
                       cudaMemcpyDeviceToHost));
      require(selected < a.vocab, "model_device_selection");
      cursor->ready_to_commit(token);
      cursor->commit(token);
      auto eos = layout->generation().is_eos(selected);
      ended = stop && eos;
      return {selected, eos};
    } catch (...) {
      poisoned = true;
      cursor->invalidate();
      throw;
    }
  }
  bool cleanup() noexcept {
    if (!host)
      return true;
    try {
      park();
      producers.clear();
      layout.reset();
      manifest.reset();
      backing.reset();
      transfer.clear();
      for (auto [d, a] : gpu) {
        if (!a.touched)
          continue;
        require(resources->snapshot().used.at(d + 1) == options.device_headroom,
                "context_has_other_device_owner");
        current(d);
        check(cudaDeviceReset());
      }
      release(host, host_loaded);
      return true;
    } catch (...) {
      poisoned = true;
      return false;
    }
  }
};
AutoModel::AutoModel(const char *root, ModelOptions o,
                     const std::map<int, std::size_t> &budgets,
                     std::shared_ptr<Resources> r,
                     const std::atomic_bool *cancel) {
  require(o.split_residency, "automatic_requires_resident");
  impl_ = std::make_unique<Impl>(o, std::move(r));
  impl_->cancel = cancel;
  try {
    impl_->initialize(root, budgets);
  } catch (...) {
    if (!impl_->cleanup())
      throw DeviceBufferQuarantine("automatic_model_cleanup_unconfirmed");
    throw;
  }
}
AutoModel::~AutoModel() {
  if (impl_)
    impl_->cleanup();
}
void AutoModel::set_cancel(const std::atomic_bool &cancel) {
  impl_->cancel = &cancel;
}
void AutoModel::begin_request() { impl_->begin_request(); }
void AutoModel::end_request() { impl_->end_request(); }
void AutoModel::park() { impl_->park(); }
void AutoModel::close() {
  if (!impl_)
    return;
  if (!impl_->cleanup())
    throw std::runtime_error("automatic_model_cleanup_unconfirmed");
  impl_.reset();
}
void AutoModel::reset() {
  require(!impl_->poisoned && impl_->state, "model_unavailable");
  impl_->reset_state();
}
stack::Selection AutoModel::step(unsigned input, bool stop) {
  return impl_->step(input, stop);
}
std::size_t AutoModel::tokens() const {
  return impl_->cursor->committed_tokens();
}
serving::WeightBacking::Stats AutoModel::cache_stats() const {
  return impl_->backing->stats();
}
} // namespace kadan::cuda
