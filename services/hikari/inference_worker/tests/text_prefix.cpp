// Token bookkeeping only; no checkpoint, forward pass or device execution.
#include "kadan/text_prefix.hpp"

#include <array>
#include <cassert>
#include <iostream>

int main() {
    using kadan::text::Prefix;
    const std::array<unsigned, 2> prompt{1, 2};
    const std::array<unsigned, 1> decoded{3};
    const std::array<unsigned, 4> extension{1, 2, 3, 4};
    Prefix prefix(8);
    assert(prefix.begin(prompt, "conversation", true) == 0);
    prefix.commit(prompt);
    prefix.commit(decoded);  // A selected but unconsumed output is not committed.
    assert(prefix.stored() == 0);
    prefix.complete();
    assert(prefix.stored() == 3);
    const auto reused = prefix.begin(extension, "conversation", true);
    assert(reused == 3 && prefix.reason() == "hit");
    const auto suffix = std::span(extension).subspan(reused);
    assert(suffix.size() == 1 && suffix[0] == 4);
    prefix.commit(suffix);
    prefix.complete();
    assert(prefix.stored() == 4);
    // Recurrence cannot be rewound, and equal input has no cached logits.
    assert(prefix.begin(extension, "conversation", true) == 0);
    assert(prefix.reason() == "miss");
    prefix.commit(extension);
    prefix.complete();
    assert(prefix.begin(prompt, "conversation", true) == 0);
    prefix.commit(prompt);
    prefix.complete();
    const std::array<unsigned, 3> edited{1, 8, 9};
    assert(prefix.begin(edited, "conversation", true) == 0);
    prefix.commit(edited);
    prefix.complete();
    assert(prefix.begin(extension, "other", true) == 0);
    prefix.commit(extension);
    prefix.complete();
    assert(prefix.begin(extension, "other", false) == 0);
    assert(prefix.reason() == "disabled" && !prefix.retaining());
    prefix.commit(extension);
    prefix.complete();
    assert(prefix.stored() == 0);
    assert(prefix.begin(prompt, "", true) == 0);
    prefix.commit(prompt);
    prefix.complete();
    assert(prefix.stored() == 0);
    // An interrupted request never publishes a valid entry.
    assert(prefix.begin(prompt, "conversation", true) == 0);
    prefix.commit(prompt);
    assert(prefix.begin(extension, "conversation", true) == 0);
    prefix.commit(extension);
    prefix.complete();
    prefix.clear();
    assert(prefix.begin(extension, "conversation", true) == 0);
    bool rejected = false;
    try {
        const std::array<unsigned, 9> too_large{};
        prefix.commit(too_large);
    } catch (const std::length_error&) {
        rejected = true;
    }
    assert(rejected && prefix.stored() == 0);
    // A checkpoint's leading tokens seed a miss and become the stored prefix.
    const std::array<unsigned, 5> replaced{1, 2, 7, 8, 9};
    assert(prefix.begin(replaced, "conversation", true) == 0);
    prefix.resume(std::span(replaced).first(2));
    assert(prefix.reason() == "checkpoint_hit");
    assert(prefix.committed().size() == 2 && prefix.conversation() == "conversation");
    prefix.commit(std::span(replaced).subspan(2));
    prefix.complete();
    assert(prefix.stored() == 5);

    // Multi-slot CheckpointPool: preserves chat and computer conversations
    // independently, shares static prefixes across fresh conversations, enforces
    // image fences, and respects strict host memory budget via LRU eviction.
    using kadan::text::AttentionState;
    using kadan::text::CheckpointPool;
    using kadan::text::ImageTag;
    using kadan::text::ModelState;
    auto dummy_state = [](std::size_t floats) {
        ModelState s;
        s.layers.emplace_back(
            AttentionState{std::vector<float>(floats, 1.f), std::vector<float>(floats, 2.f)}
        );
        return s;
    };

    const std::size_t one_entry_bytes =
        kadan::text::state_memory_bytes(dummy_state(16)) + 1024;
    CheckpointPool pool(one_entry_bytes * 3);

    const std::array<unsigned, 3> sys_prefix{10, 11, 12};
    const std::array<unsigned, 5> chat_turn1{10, 11, 12, 20, 21};
    const std::array<unsigned, 7> chat_turn2{10, 11, 12, 20, 21, 22, 23};
    const std::array<unsigned, 5> comp_turn1{10, 11, 12, 30, 31};
    const std::array<unsigned, 7> comp_turn2{10, 11, 12, 30, 31, 32, 33};
    const std::array<unsigned, 5> fresh_chat{10, 11, 12, 99, 100};

    assert(pool.save("", true, sys_prefix, {}, dummy_state(16)));
    assert(pool.save("chat-1", false, chat_turn1, {}, dummy_state(16)));
    assert(pool.save("computer-1", false, comp_turn1, {}, dummy_state(16)));
    assert(pool.size() == 3 && pool.used_bytes() <= pool.budget_bytes());

    // Interleaving chat and computer hits each conversation's own longest prefix.
    const auto* hit_chat = pool.find_best(chat_turn2, "chat-1");
    assert(hit_chat && !hit_chat->is_static && hit_chat->tokens.size() == 5);
    const auto* hit_comp = pool.find_best(comp_turn2, "computer-1");
    assert(hit_comp && !hit_comp->is_static && hit_comp->tokens.size() == 5);

    // Fresh conversation after clear chat hits the shared static prefix.
    const auto* hit_fresh = pool.find_best(fresh_chat, "chat-2");
    assert(hit_fresh && hit_fresh->is_static && hit_fresh->tokens.size() == 3);

    // Image tags fence reuse: mismatched hash rejects conversation entry and
    // falls back to image-free static prefix.
    const std::array<ImageTag, 1> img_a{ImageTag{3, 2, 0xAAAA}};
    const std::array<ImageTag, 1> img_b{ImageTag{3, 2, 0xBBBB}};
    assert(pool.save("chat-img", false, chat_turn1, img_a, dummy_state(16)));
    assert(pool.used_bytes() <= pool.budget_bytes());
    const auto* same_img = pool.find_best(chat_turn2, "chat-img", img_a);
    assert(same_img && !same_img->is_static && same_img->tokens.size() == 5);
    const auto* diff_img = pool.find_best(chat_turn2, "chat-img", img_b);
    assert(diff_img && diff_img->is_static && diff_img->tokens.size() == 3);
    // An image overlapping the static prefix disables even static reuse.
    const std::array<ImageTag, 1> img_in_sys{ImageTag{1, 2, 0xCCCC}};
    assert(pool.find_best(fresh_chat, "chat-3", img_in_sys) == nullptr);

    // Oversized entry is rejected without evicting existing entries.
    const auto before_count = pool.size();
    assert(!pool.save("huge", false, chat_turn1, {}, dummy_state(100000)));
    assert(pool.size() == before_count);

    // Prefix::resume supports custom static_prefix_hit reason.
    assert(prefix.begin(fresh_chat, "chat-2", true) == 0);
    prefix.resume(sys_prefix, "static_prefix_hit");
    assert(prefix.reason() == "static_prefix_hit");

    std::cout << "Resident prefix and checkpoint pool units passed; no inference\n";
}
