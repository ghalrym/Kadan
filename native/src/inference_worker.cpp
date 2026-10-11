#include "kadan/inference_engines.hpp"
#include <chrono>
#include <csignal>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <poll.h>
#include <sys/file.h>
#include <thread>
#include <unistd.h>

namespace kadan::serving {
namespace {
std::atomic_bool stopping = false;
void stop_signal(int) { stopping.store(true); }

class PhysicalMemory {
public:
  PhysicalMemory() {
    library_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    require(library_, "nvml_unavailable");
    initialize_ = symbol<int (*)()>("nvmlInit_v2");
    shutdown_ = symbol<int (*)()>("nvmlShutdown");
    count_ = symbol<int (*)(unsigned *)>("nvmlDeviceGetCount_v2");
    device_ =
        symbol<int (*)(unsigned, void **)>("nvmlDeviceGetHandleByIndex_v2");
    memory_ = symbol<int (*)(void *, Memory *)>("nvmlDeviceGetMemoryInfo");
    require(initialize_() == 0 && count_(&devices_) == 0 &&
                devices_ <= Resources::max_devices,
            "nvml_initialize");
  }
  ~PhysicalMemory() {
    if (shutdown_)
      shutdown_();
    if (library_)
      dlclose(library_);
  }
  Footprint probe() const {
    Footprint available(devices_ + 1);
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
      if (line.starts_with("MemAvailable:")) {
        available[0] = std::stoull(line.substr(13)) * 1024;
        break;
      }
    }
    require(available[0] > 0, "host_memory_probe");
    std::ifstream limit_file("/sys/fs/cgroup/memory.max"),
        current_file("/sys/fs/cgroup/memory.current");
    std::string limit;
    Bytes current = 0;
    if (limit_file >> limit && current_file >> current && limit != "max") {
      const auto maximum = std::stoull(limit);
      available[0] = std::min<Bytes>(available[0],
                                     maximum > current ? maximum - current : 0);
    }
    constexpr Bytes host_margin = 1024ULL * 1024 * 1024;
    available[0] = available[0] > host_margin ? available[0] - host_margin : 0;
    for (unsigned index = 0; index < devices_; ++index) {
      void *device = nullptr;
      Memory memory{};
      require(device_(index, &device) == 0 && memory_(device, &memory) == 0,
              "device_memory_probe");
      constexpr Bytes margin = 512ULL * 1024 * 1024;
      available[index + 1] = memory.free > margin ? memory.free - margin : 0;
    }
    return available;
  }

private:
  struct Memory {
    unsigned long long total, free, used;
  };
  template <class Function> Function symbol(const char *name) {
    auto function = reinterpret_cast<Function>(dlsym(library_, name));
    require(function, "nvml_symbol");
    return function;
  }
  void *library_ = nullptr;
  int (*initialize_)() = nullptr;
  int (*shutdown_)() = nullptr;
  int (*count_)(unsigned *) = nullptr;
  int (*device_)(unsigned, void **) = nullptr;
  int (*memory_)(void *, Memory *) = nullptr;
  unsigned devices_ = 0;
};

class Output {
public:
  void send(const Json &message) {
    const auto line = message.dump() + '\n';
    require(line.size() <= 1024 * 1024, "worker_output_bound");
    std::lock_guard guard(mutex_);
    std::size_t offset = 0;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (offset < line.size()) {
      require(std::chrono::steady_clock::now() < deadline,
              "worker_output_timeout");
      pollfd descriptor{STDOUT_FILENO, POLLOUT, 0};
      const auto ready = poll(&descriptor, 1, 100);
      if (ready < 0 && errno == EINTR)
        continue;
      require(ready >= 0 &&
                  !(descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)),
              "worker_output_closed");
      if (!ready)
        continue;
      const auto written =
          write(STDOUT_FILENO, line.data() + offset, line.size() - offset);
      if (written > 0)
        offset += written;
      else
        require(errno == EINTR || errno == EAGAIN, "worker_output_write");
    }
  }

private:
  std::mutex mutex_;
};

const char *state_name(JobState state) {
  switch (state) {
  case JobState::preparing: return "preparing";
  case JobState::loading:
    return "loading";
  case JobState::running:
    return "running";
  case JobState::waiting_for_resources:
    return "waiting_for_resources";
  case JobState::succeeded:
    return "succeeded";
  case JobState::failed:
    return "failed";
  case JobState::cancelled:
    return "cancelled";
  }
  throw std::invalid_argument("job_state");
}
Operation operation(const std::string &name) {
  if (name == "generate")
    return Operation::generate;
  if (name == "completion")
    return Operation::completion;
  if (name == "load")
    return Operation::load;
  if (name == "unload")
    return Operation::unload;
  throw std::invalid_argument("request_operation");
}

Workload workload(const std::string &name) {
  if (name == "llm")
    return Workload::llm;
  if (name == "image")
    return Workload::image;
  if (name == "video")
    return Workload::video;
  if (name == "stt")
    return Workload::speech;
  if (name == "tts")
    return Workload::tts;
  if (name == "decisions")
    return Workload::decision;
  throw std::invalid_argument("unsupported_workload");
}

void read_requests(InferenceOwner &owner, Output &output) {
  std::string line;
  try {
    while (!stopping.load()) {
      pollfd descriptor{STDIN_FILENO, POLLIN, 0};
      auto ready = poll(&descriptor, 1, 100);
      if (ready < 0 && errno == EINTR)
        continue;
      require(ready >= 0, "worker_input_poll");
      if (!ready)
        continue;
      char byte;
      const auto count = read(STDIN_FILENO, &byte, 1);
      if (count == 0)
        break;
      if (count < 0 && errno == EINTR)
        continue;
      require(count == 1, "worker_input_read");
      if (byte != '\n') {
        require(line.size() < 256 * 1024, "worker_input_bound");
        line += byte;
        continue;
      }
      std::string id;
      try {
        const auto request = Json::parse(line);
        id = request.at("id");
        require(id.size() == 32 && id.find_first_not_of("0123456789abcdef") ==
                                       std::string::npos,
                "request_id");
        const auto command = request.at("command").get<std::string>();
        if (command == "submit") {
          owner.submit({id, request.at("model"),
                        operation(request.at("operation")), "",
                        workload(request.at("feature")), false});
          output.send({{"id", id}, {"state", "queued"}});
        } else if (command == "prepared") {
          const auto &payload = request.at("payload");
          const auto configuration =
              Json{{"checkpoint", payload.value("checkpoint", "")},
                   {"context_limit", payload.value("context_limit", 0u)}}
                  .dump();
          owner.prepared(id, payload.dump(), request.value("error", ""),
                         configuration);
        } else if (command == "cancel") {
          owner.cancel(id);
        } else if (command == "stop") {
          stopping.store(true);
        } else
          throw std::invalid_argument("worker_command");
      } catch (const std::exception &error) {
        output.send(
            {{"id", id}, {"state", "rejected"}, {"error", error.what()}});
      }
      line.clear();
    }
  } catch (const std::exception &error) {
    std::cerr << "worker transport: " << error.what() << '\n';
  }
  stopping.store(true);
  owner.stop();
}
} // namespace

int serve(const char *lock_path) {
  int lock = open(lock_path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  require(lock >= 0 && flock(lock, LOCK_EX | LOCK_NB) == 0,
          "inference_worker_already_running");
  PhysicalMemory memory;
  Output output;
  std::string active;
  auto progress = [&](const char *phase, std::size_t value) {
    output.send({{"id", active},
                 {"state", "progress"},
                 {"phase", phase},
                 {"value", value}});
  };
  auto content = [&](const std::string &text) {
    output.send({{"id", active}, {"state", "content"}, {"content", text}});
  };
  auto factory = [&](Workload kind, std::shared_ptr<Resources> resources) {
    switch (kind) {
    case Workload::llm:
      return chat_engine(resources, progress, content);
    case Workload::image:
      return image_engine(resources, progress);
    case Workload::video:
      return video_engine(resources, progress);
    case Workload::speech:
      return whisper_engine(resources, progress);
    case Workload::tts:
      return speech_engine(resources, progress);
    case Workload::decision:
      return decision_engine(resources, progress);
    }
    throw std::invalid_argument("unsupported_workload");
  };
  InferenceOwner *owner_pointer = nullptr;
  auto publish = [&](const std::string &id, JobState state,
                     const std::string &result) {
    active = id;
    Json message{{"id", id}, {"state", state_name(state)}};
    if (state == JobState::succeeded)
      message["result"] = Json::parse(result);
    else if (state == JobState::failed)
      message["error"] = result;
    const auto snapshot = owner_pointer->snapshot();
    message["memory"] = {{"capacity", snapshot.capacity},
                         {"used", snapshot.used},
                         {"residents", snapshot.residents}};
    output.send(message);
  };
  auto capacity = memory.probe();
  // The ceiling is the machine shape, not its startup free memory. Each new
  // engine obtains a fresh physical measurement after the old engine closes.
  std::fill(capacity.begin(), capacity.end(),
            std::numeric_limits<Bytes>::max());
  InferenceOwner owner(
      capacity, factory, [&] { return memory.probe(); }, publish);
  owner_pointer = &owner;
  output.send({{"state", "ready"}, {"protocol", 1}, {"pid", getpid()}});
  std::thread reader([&] { read_requests(owner, output); });
  try {
    while (!stopping.load()) {
      if (!owner.advance())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    owner.shutdown();
  } catch (...) {
    stopping.store(true);
    owner.stop();
    reader.join();
    throw;
  }
  reader.join();
  close(lock);
  return 0;
}
} // namespace kadan::serving

int main(int argc, char **argv) {
  std::signal(SIGTERM, kadan::serving::stop_signal);
  std::signal(SIGINT, kadan::serving::stop_signal);
  std::signal(SIGPIPE, SIG_IGN);
  try {
    kadan::serving::require(argc == 2,
                            "usage: kadan-inference-worker LOCK_PATH");
    const auto flags = fcntl(STDOUT_FILENO, F_GETFL);
    kadan::serving::require(
        flags >= 0 && fcntl(STDOUT_FILENO, F_SETFL, flags | O_NONBLOCK) == 0,
        "stdout_nonblocking");
    return kadan::serving::serve(argv[1]);
  } catch (const std::exception &error) {
    std::cerr << "inference_worker: " << error.what() << '\n';
    return 1;
  }
}
