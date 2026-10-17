#include "kadan/image_input.hpp"

#include <cassert>
#include <stdexcept>

using namespace kadan::image;

namespace {
template <typename F> void rejects(F&& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return;
    }
    assert(false);
}

ImagePrompt placeholder() {
    ImagePrompt prompt;
    prompt.tokens[0] = 151652;
    prompt.tokens[1] = 151655;
    prompt.tokens[2] = 151653;
    prompt.tokens[3] = 1234;
    prompt.count = 4;
    return prompt;
}
}  // namespace

int main() {
    auto prompt = placeholder();
    expand_image_prompt(prompt, 96, 64);
    assert(prompt.count == 9 && prompt.image_start == 1 && prompt.image_count == 6);
    assert(prompt.grid_h == 2 && prompt.grid_w == 3);
    assert(
        prompt.tokens[0] == 151652 && prompt.tokens[7] == 151653
        && prompt.tokens[8] == 1234
    );
    for (int i = 1; i < 7; ++i)
        assert(prompt.tokens[i] == 151655);
    const auto positions = image_text_positions(prompt);
    assert((positions[0] == std::vector<std::int64_t>{0, 1, 1, 1, 1, 1, 1, 4, 5}));
    assert((positions[1] == std::vector<std::int64_t>{0, 1, 1, 1, 2, 2, 2, 4, 5}));
    assert((positions[2] == std::vector<std::int64_t>{0, 1, 2, 3, 1, 2, 3, 4, 5}));
    ImagePrompt text;
    text.count = 4;
    for (const auto& axis : image_text_positions(text))
        assert((axis == std::vector<std::int64_t>{0, 1, 2, 3}));
    rejects([] {
        auto p = placeholder();
        expand_image_prompt(p, 0, 64);
    });
    rejects([] {
        auto p = placeholder();
        expand_image_prompt(p, 48, 64);
    });
    rejects([] {
        auto p = placeholder();
        p.tokens[1] = 123;
        expand_image_prompt(p, 64, 64);
    });
    rejects([] {
        auto p = placeholder();
        p.tokens[3] = 151655;
        expand_image_prompt(p, 64, 64);
    });
    rejects([] {
        auto p = placeholder();
        expand_image_prompt(p, 3072, 3072);
    });
    rejects([] {
        auto p = placeholder();
        p.count = p.tokens.size() + 1;
        expand_image_prompt(p, 64, 64);
    });
}
