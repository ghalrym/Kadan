#pragma once
#include <cstddef>
#include <functional>
#include <string_view>
#include <utility>

namespace kadan::video {
// RoPE reports every head. Keep that diagnostic granularity inside the engine,
// but batch transport updates; stage transitions retain the last exact count.
class VideoProgress {
public:
    using Sink = std::function<void(const char *, std::size_t)>;
    explicit VideoProgress(Sink sink) : sink_(std::move(sink)) {}

    void reset() { pending_ = sent_ = 0; }
    void update(const char *phase, std::size_t count) {
        if (std::string_view(phase) != "rope") {
            flush();
            sink_(phase, count);
            return;
        }
        if (count < pending_) flush();
        pending_ = count;
        if (!sent_ || count - sent_ >= 4096) {
            sink_("rope", count);
            sent_ = count;
        }
    }
private:
    void flush() {
        if (pending_ != sent_) sink_("rope", pending_);
        pending_ = sent_ = 0;
    }
    Sink sink_;
    std::size_t pending_ = 0, sent_ = 0;
};
}
