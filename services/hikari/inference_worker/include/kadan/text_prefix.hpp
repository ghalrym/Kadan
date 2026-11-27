#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kadan/text_state.hpp"

namespace kadan::text {
struct ImageTag {
    std::size_t offset = 0;
    std::size_t tokens = 0;
    std::uint64_t hash = 0;
    bool operator==(const ImageTag&) const = default;
};

// Returns true iff every image overlapping [0, prefix_len) lies strictly inside
// [0, prefix_len) and matches identically between cached and current prompts.
inline bool images_match_prefix(
    std::span<const ImageTag> cached,
    std::span<const ImageTag> current,
    std::size_t prefix_len
) {
    std::size_t ci = 0, pi = 0;
    while (ci < cached.size() && cached[ci].offset < prefix_len) {
        if (cached[ci].offset + cached[ci].tokens > prefix_len)
            return false;
        if (pi >= current.size() || current[pi].offset >= prefix_len)
            return false;
        if (current[pi].offset + current[pi].tokens > prefix_len)
            return false;
        if (cached[ci] != current[pi])
            return false;
        ++ci;
        ++pi;
    }
    if (pi < current.size() && current[pi].offset < prefix_len)
        return false;
    return true;
}

inline std::vector<ImageTag>
images_within_prefix(std::span<const ImageTag> images, std::size_t prefix_len) {
    std::vector<ImageTag> out;
    for (const auto& img : images) {
        if (img.offset >= prefix_len)
            break;
        if (img.offset + img.tokens <= prefix_len)
            out.push_back(img);
    }
    return out;
}

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

    // Continues a miss from a checkpoint of an earlier prompt's leading tokens.
    void resume(
        std::span<const unsigned> tokens,
        std::string_view reason = "checkpoint_hit"
    ) {
        commit(tokens);
        reason_ = reason;
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

    // Tokens committed by the current request so far.
    std::span<const unsigned> committed() const {
        return tokens_;
    }

    std::string_view conversation() const {
        return conversation_;
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

// Bounded LRU pool of compacted model checkpoints. Holds both per-conversation
// checkpoints (live snapshots on conversation switch and leading-turn boundaries)
// and cross-conversation static prefixes (e.g. identical system prompt + tools
// with zero images). Total retained memory never exceeds `budget_bytes`.
class CheckpointPool {
public:
    struct Entry {
        std::string conversation;
        bool is_static = false;
        std::vector<unsigned> tokens;
        std::vector<ImageTag> images;
        ModelState state;
        std::size_t bytes = 0;
        std::uint64_t last_used = 0;
    };

    explicit CheckpointPool(std::size_t budget_bytes = 0)
        : budget_bytes_(budget_bytes) {
    }

    void clear() {
        entries_.clear();
        used_bytes_ = 0;
    }

    void erase(std::string_view conversation) {
        if (conversation.empty())
            return;
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (!it->is_static && it->conversation == conversation) {
                used_bytes_ -= it->bytes;
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::size_t budget_bytes() const {
        return budget_bytes_;
    }

    std::size_t used_bytes() const {
        return used_bytes_;
    }

    std::size_t size() const {
        return entries_.size();
    }

    bool save(
        std::string_view conversation,
        bool is_static,
        std::span<const unsigned> tokens,
        std::span<const ImageTag> images,
        ModelState state
    ) {
        if (!budget_bytes_ || tokens.empty())
            return false;
        if (!is_static && conversation.empty())
            return false;
        if (is_static && !images.empty())
            return false;

        Entry entry;
        entry.conversation = is_static ? std::string() : std::string(conversation);
        entry.is_static = is_static;
        entry.tokens.assign(tokens.begin(), tokens.end());
        entry.images = images_within_prefix(images, tokens.size());
        entry.state = std::move(state);
        entry.bytes = state_memory_bytes(entry.state)
                      + entry.tokens.capacity() * sizeof(unsigned)
                      + entry.images.capacity() * sizeof(ImageTag)
                      + entry.conversation.capacity();
        entry.last_used = ++clock_;

        if (entry.bytes > budget_bytes_)
            return false;

        // Deduplicate exact matches (same scope + tokens + images).
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->is_static == entry.is_static
                && it->conversation == entry.conversation
                && it->tokens == entry.tokens
                && it->images == entry.images) {
                used_bytes_ -= it->bytes;
                entries_.erase(it);
                break;
            }
        }

        // Keep at most 2 checkpoints per conversation (boundary + live) so one
        // conversation does not evict other active conversations unnecessarily.
        if (!is_static) {
            std::size_t count = 0;
            auto oldest = entries_.end();
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                if (!it->is_static && it->conversation == entry.conversation) {
                    ++count;
                    if (oldest == entries_.end() || it->last_used < oldest->last_used)
                        oldest = it;
                }
            }
            if (count >= 2 && oldest != entries_.end()) {
                used_bytes_ -= oldest->bytes;
                entries_.erase(oldest);
            }
        }

        while (used_bytes_ + entry.bytes > budget_bytes_ && !entries_.empty()) {
            auto victim = std::min_element(
                entries_.begin(),
                entries_.end(),
                [](const Entry& a, const Entry& b) {
                    return a.last_used < b.last_used;
                }
            );
            used_bytes_ -= victim->bytes;
            entries_.erase(victim);
        }

        if (used_bytes_ + entry.bytes > budget_bytes_)
            return false;

        used_bytes_ += entry.bytes;
        entries_.push_back(std::move(entry));
        return true;
    }

    const Entry* find_best(
        std::span<const unsigned> prompt,
        std::string_view conversation,
        std::span<const ImageTag> prompt_images = {}
    ) {
        Entry* best = nullptr;
        for (auto& entry : entries_) {
            if (entry.tokens.size() >= prompt.size())
                continue;
            if (!std::equal(entry.tokens.begin(), entry.tokens.end(), prompt.begin()))
                continue;
            if (entry.is_static) {
                if (!images_match_prefix({}, prompt_images, entry.tokens.size()))
                    continue;
            } else {
                if (conversation.empty() || entry.conversation != conversation)
                    continue;
                if (!images_match_prefix(
                        entry.images, prompt_images, entry.tokens.size()
                    ))
                    continue;
            }
            if (!best || entry.tokens.size() > best->tokens.size()
                || (entry.tokens.size() == best->tokens.size() && !entry.is_static
                    && best->is_static))
                best = &entry;
        }
        if (best)
            best->last_used = ++clock_;
        return best;
    }

private:
    std::size_t budget_bytes_ = 0;
    std::size_t used_bytes_ = 0;
    std::uint64_t clock_ = 0;
    std::vector<Entry> entries_;
};
}  // namespace kadan::text
