#pragma once
#include <atomic>
#include <map>
#include <memory>

#include "kadan/model_worker.hpp"
#include "kadan/resources.hpp"

namespace kadan::cuda {
// Text-only GPT-OSS 20B/120B with packed MXFP4 experts. Complete layers run
// on one device each; experts that do not fit stay in mapped host memory.
// Owns its process's CUDA contexts exclusively, like AutoModel.
class GptOssModel {
public:
    static constexpr std::size_t device_headroom = 512ULL * 1024 * 1024;

    GptOssModel(
        const char* root,
        std::size_t capacity,
        const std::map<int, std::size_t>& budgets,
        std::shared_ptr<Resources>,
        const std::atomic_bool* cancel = nullptr
    );
    ~GptOssModel();
    GptOssModel(const GptOssModel&) = delete;
    GptOssModel& operator=(const GptOssModel&) = delete;
    // Host bytes reserved at construction: metadata, staging, the mapped
    // embedding and every expert that the placement leaves on the host.
    static std::size_t host_bytes(
        const char* root, std::size_t capacity, const std::map<int, std::size_t>&
    );
    void set_cancel(const std::atomic_bool&);
    // Requests allocate context state; the first after park reloads weights.
    void begin_request();
    void end_request();
    void park();
    void close();
    // Logits are computed only when requested, such as after the last prompt
    // token; otherwise the returned id is meaningless.
    serving::Token step(unsigned token, bool logits);
    std::size_t tokens() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace kadan::cuda
