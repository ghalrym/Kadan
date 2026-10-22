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
    std::cout << "Resident prefix bookkeeping units passed; no inference\n";
}
