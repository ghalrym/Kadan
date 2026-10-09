#pragma once
#include "kadan/resources.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
namespace kadan::decision {
// Complete CPU Laya execution. Serialized owner; only cancellation is concurrent.
// Supported profile: ModernBERT/F16 + default RoPE, byte BPE/NFC, two-action
// Laya head; at most 512 input tokens, 8 questions and 64 options per question.
// Admission is a conservative allocation envelope, not an operating-system RSS cap.
// Own mode accounts decoded weights plus 128 MiB metadata and 256 MiB execution
// scratch. FIFO mode borrows one 3 GiB envelope for those same allocations.
// Caller admits request/output strings (each <= 64 KiB) until publication.
class Executor {
  public:
    using Observer = std::function<void(std::string_view)>;
    explicit Executor(std::shared_ptr<Resources>);
    ~Executor();
    Executor(const Executor &) = delete;
    // An optional FIFO-owned envelope covers model, scratch and publication.
    // The queue, not this executor, acknowledges and releases that envelope.
    static constexpr Bytes envelope_bytes = 3ULL * 1024 * 1024 * 1024;
    void load(const std::string &root, const std::atomic_bool &cancel, Handle admission = 0,
              const Observer &observer = {});
    std::string execute(std::string_view request, const std::atomic_bool &cancel,
                        const Observer &observer = {});
    void unload();
    bool loaded() const;

  private:
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::unique_ptr<Impl> impl_;
    Handle resident_ = 0;
    bool borrowed_ = false;
};
} // namespace kadan::decision
