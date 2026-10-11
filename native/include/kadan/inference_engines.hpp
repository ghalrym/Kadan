#pragma once
#include "kadan/inference_owner.hpp"
#include "kadan/worker_compute.hpp"
#include <nlohmann/json.hpp>
namespace kadan::serving {
using Json = nlohmann::json;
using Progress = std::function<void(const char *, std::size_t)>;
inline void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
inline void check_cancel(const std::atomic_bool &cancel) {
  require(!cancel.load(), "request_cancelled");
}
inline ComputePlan media_plan(const Footprint &available,
                              Bytes minimum = compute_context_bytes +
                                              2ULL * 1024 * 1024 * 1024) {
  ComputePlan plan{available, {}};
  for (std::size_t i = 1; i < available.size(); ++i)
    if (available[i] >= minimum)
      plan.devices.push_back(int(i - 1));
  if (plan.devices.empty())
    throw AdmissionWait("insufficient_device_headroom");
  return plan;
}
class HostAllocation {
public:
  HostAllocation(std::shared_ptr<Resources> resources, Workload kind,
                 Bytes bytes)
      : resources_(std::move(resources)),
        handle_(resources_->reserve(kind, host_footprint(*resources_, bytes))) {
  }
  ~HostAllocation() {
    if (handle_)
      resources_->released(handle_);
  }
  HostAllocation(const HostAllocation &) = delete;

private:
  std::shared_ptr<Resources> resources_;
  Handle handle_;
};
std::unique_ptr<InferenceEngine> decision_engine(std::shared_ptr<Resources>,
                                                 Progress);
std::unique_ptr<InferenceEngine> image_engine(std::shared_ptr<Resources>,
                                              Progress);
std::unique_ptr<InferenceEngine> video_engine(std::shared_ptr<Resources>,
                                              Progress);
std::unique_ptr<InferenceEngine> whisper_engine(std::shared_ptr<Resources>,
                                                Progress);
std::unique_ptr<InferenceEngine> speech_engine(std::shared_ptr<Resources>,
                                               Progress);
std::unique_ptr<InferenceEngine>
    chat_engine(std::shared_ptr<Resources>, Progress,
                std::function<void(const std::string &)>);
} // namespace kadan::serving
