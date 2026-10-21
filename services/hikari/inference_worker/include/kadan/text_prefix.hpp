#pragma once
#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace kadan::text {
// Hybrid recurrent models can continue their complete committed state, but
// cannot rewind it to an arbitrary common prefix without a snapshot.
class Prefix {
public:
    explicit Prefix(std::size_t capacity = 0) : capacity_(capacity) {
        tokens_.reserve(capacity);
    }

    std::size_t
    begin(std::span<const unsigned> prompt, std::string_view conversation, bool reuse) {
        const bool hit = valid_ && reuse && !conversation.empty()
                         && conversation == conversation_
                         && tokens_.size() < prompt.size()
                         && std::equal(tokens_.begin(), tokens_.end(), prompt.begin());
        reason_ = !reuse ? "disabled" : hit ? "hit" : "miss";
        const auto reused = hit ? tokens_.size() : 0;
        if (!hit)
            tokens_.clear();
        conversation_ = conversation;
        retain_ = reuse && !conversation.empty();
        valid_ = false;
        return reused;
    }

    void commit(std::span<const unsigned> tokens) {
        if (!retain_)
            return;
        if (tokens.size() > capacity_ - tokens_.size())
            throw std::length_error("text_prefix_capacity");
        tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
    }

    void complete() {
        valid_ = retain_ && !tokens_.empty();
    }

    void clear() {
        tokens_.clear();
        conversation_.clear();
        retain_ = valid_ = false;
    }

    std::size_t memory_bytes() const {
        return tokens_.capacity() * sizeof(unsigned) + conversation_.capacity();
    }

    bool retaining() const {
        return retain_;
    }

    std::size_t stored() const {
        return valid_ ? tokens_.size() : 0;
    }

    std::string_view reason() const {
        return reason_;
    }

private:
    std::size_t capacity_;
    std::vector<unsigned> tokens_;
    std::string conversation_;
    std::string_view reason_ = "miss";
    bool retain_ = false, valid_ = false;
};
}  // namespace kadan::text
