#pragma once
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <string_view>

#include "kadan/dense_compute.hpp"
#include "kadan/device_placement.hpp"
#include "kadan/media_session.hpp"

namespace kadan {
inline Footprint host_footprint(Resources& r, Bytes bytes) {
    auto f = r.snapshot().capacity;
    std::fill(f.begin(), f.end(), 0);
    f[0] = bytes;
    return f;
}

struct ComputePlan {
    Footprint capacity;
    std::vector<int> devices;
};

inline ComputePlan compute_plan(
    Bytes host,
    std::string_view selection,
    std::string_view budget,
    bool available,
    std::string_view indexed = {}
) {
    if (!host)
        throw std::invalid_argument("compute_host_budget");
    ComputePlan p{{host}, {}};
    if (selection.empty())
        return p;
    if (!available)
        throw std::invalid_argument("compute_cuda_not_built");
    p.devices = compute_devices(selection);
    auto budgets = device_budgets(
        p.devices, budget, indexed, compute_context_bytes + 16 * 1024 * 1024
    );
    p.capacity.resize(budgets.size());
    for (int device : p.devices)
        p.capacity[device + 1] = budgets[device + 1];
    return p;
}

inline ComputePlan worker_compute_plan(Bytes host, const char* selection_name) {
    const char* s = std::getenv(selection_name);
    const char* b = std::getenv("KADAN_NATIVE_GPU_BUDGET_BYTES");
#ifdef KADAN_WORKER_CUDA
    constexpr bool available = true;
#else
    constexpr bool available = false;
#endif
    const char* indexed = std::getenv("KADAN_NATIVE_GPU_BUDGETS");
    return compute_plan(
        host, s ? s : "", b ? b : "", available, indexed ? indexed : ""
    );
}

struct WorkerCompute {
    std::shared_ptr<Resources> resources;
    std::shared_ptr<DenseCompute> compute;
    std::shared_ptr<MediaSessionActivity> media_activity =
        std::make_shared<MediaSessionActivity>();
    Footprint context;
    Handle context_handle = 0;
    Workload workload;
    std::vector<int> devices;

    WorkerCompute(
        const ComputePlan& plan, Workload kind, std::shared_ptr<Resources> owner = {}
    )
        : resources(
              owner ? std::move(owner) : std::make_shared<Resources>(plan.capacity)
          ),
          context(plan.capacity.size()), workload(kind), devices(plan.devices) {
#ifdef KADAN_WORKER_CUDA
        if (!plan.devices.empty()) {
            for (int d : plan.devices)
                context[d + 1] = compute_context_bytes;
            // Keep the matching parent reservation until acknowledged park destroys
            // these contexts, or the child has been reaped during cancellation.
            context_handle = resources->reserve(workload, context);
            try {
                compute = cuda_dense_compute(resources, plan.devices, workload);
            } catch (...) {
                resources->released(context_handle);
                context_handle = 0;
                throw;
            }
        }
#else
        (void)workload;
        if (!plan.devices.empty())
            throw std::invalid_argument("compute_cuda_not_built");
#endif
    }

    ~WorkerCompute() noexcept {
        try {
            park();
        } catch (...) {
            // Preserve the reservation when physical cleanup cannot be confirmed.
            // The shared owner detects this nonempty ledger and stops admission.
        }
    }

    Footprint execution_footprint(Bytes host_headroom) const {
        const auto state = resources->snapshot();
        auto bound = state.capacity;
        bound[0] =
            state.used[0] + std::min(host_headroom, state.capacity[0] - state.used[0]);
        const auto device_memory = compute ? compute->execution_memory() : Footprint{};
        if (device_memory.empty())
            return bound;
        if (device_memory.size() != bound.size())
            throw std::invalid_argument("compute_execution_shape");
        for (std::size_t i = 1; i < bound.size(); ++i) {
            const auto bytes = std::min(device_memory[i], bound[i]);
            bound[i] =
                std::max(state.used[i], bytes + std::min(context[i], bound[i] - bytes));
        }
        return bound;
    }

    // An engine may own additional resident banks outside DenseCompute. They
    // remain charged while idle; park still requires their prior retirement.
    void idle(const Footprint& engine_weights = {}) const {
        if (media_activity->owners.load())
            throw std::runtime_error("compute_media_session_busy");
        if (compute)
            compute->release_scratch();
        const auto retained = compute ? compute->retained_weights() : Footprint{};
        const auto used = resources->snapshot().used;
        if (!engine_weights.empty() && engine_weights.size() != used.size())
            throw std::invalid_argument("compute_retained_shape");
        for (std::size_t i = 1; i < used.size(); ++i)
            if ((engine_weights.empty() ? 0 : engine_weights[i]) > used[i]
                || used[i] - (engine_weights.empty() ? 0 : engine_weights[i])
                       != (context_handle ? context[i] : 0)
                              + (retained.empty() ? 0 : retained.at(i)))
                throw std::runtime_error("compute_device_cleanup_unconfirmed");
    }

    void park() {
        idle();
        if (context_handle) {
            compute->release_devices();
            resources->released(context_handle);
            context_handle = 0;
        }
    }

    void resume() {
        if (compute && !context_handle)
            context_handle = resources->reserve(workload, context);
    }
};
}  // namespace kadan
