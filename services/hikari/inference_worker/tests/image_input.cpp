#include "kadan/image_input.hpp"

#include <source_location>
#include <stdexcept>
#include <vector>

namespace {
void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error("image_input_unit:" + std::to_string(at.line()));
}

template <typename F> void rejects(F&& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return;
    }
    check(false);
}
}  // namespace

int main() {
    using namespace kadan::image;
    check(
        format_image_prompt("a red fox")
        == "<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    const std::vector<std::uint32_t> short_ids{151644, 872, 198, 64};
    const auto packed = pack_image_prompt(short_ids);
    check(packed.active_count == 4);
    for (std::size_t i = 0; i < 4; ++i)
        check(packed.tokens[i] == short_ids[i]);
    for (std::size_t i = 4; i < image_prompt_length; ++i)
        check(packed.tokens[i] == image_prompt_pad_token);

    std::vector<std::uint32_t> long_ids(700);
    for (std::size_t i = 0; i < long_ids.size(); ++i)
        long_ids[i] = std::uint32_t(100 + i);
    const auto truncated = pack_image_prompt(long_ids);
    check(truncated.active_count == image_prompt_length);
    for (std::size_t i = 0; i < image_prompt_length; ++i)
        check(truncated.tokens[i] == long_ids[i]);

    rejects([] { (void)pack_image_prompt({}); });
}
