#include "kadan/image_reference.hpp"

#include <source_location>
#include <stdexcept>

namespace {
void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error("image_reference_unit:" + std::to_string(at.line()));
}

void batch_padding_keeps_requests_independent() {
    using namespace kadan::image;
    const auto short_prompt = image_reference_layout(6, 2, 1, 2, 2, 2, 4);
    const auto long_prompt = image_reference_layout(10, 2, 1, 2, 2, 2, 4);
    const auto batch = image_batch_reference_layout({short_prompt, long_prompt});
    check(batch.padded && batch.prefix == long_prompt.prefix);
    for (std::size_t i = 0; i < batch.items.size(); ++i) {
        const auto& original = i ? long_prompt : short_prompt;
        const auto& padded = batch.items[i];
        check(padded.valid_prefix == original.prefix);
        check(padded.positions.size() == long_prompt.positions.size());
        check(
            std::equal(
                original.positions.begin(),
                original.positions.begin() + original.prefix,
                padded.positions.begin()
            )
        );
        check(
            std::equal(
                original.positions.begin() + original.prefix,
                original.positions.end(),
                padded.positions.begin() + batch.prefix
            )
        );
        for (std::size_t token = 0; token < padded.valid_keys.size(); ++token)
            check(
                bool(padded.valid_keys[token])
                == (token < std::size_t(original.prefix)
                    || token >= std::size_t(batch.prefix))
            );
        check(padded.segments.size() == original.segments.size());
        for (std::size_t segment = 0; segment < original.segments.size(); ++segment) {
            check(padded.segments[segment].start == original.segments[segment].start);
            check(padded.segments[segment].end == original.segments[segment].end);
            check(padded.segments[segment].text == original.segments[segment].text);
        }
    }
    const auto singleton = image_batch_reference_layout({short_prompt});
    check(!singleton.padded && singleton.prefix == short_prompt.prefix);
    check(singleton.items[0].positions == short_prompt.positions);
    check(
        std::all_of(
            singleton.items[0].valid_keys.begin(),
            singleton.items[0].valid_keys.end(),
            [](auto value) { return value == 1; }
        )
    );
    bool rejected = false;
    try {
        (void)image_batch_reference_layout(
            {short_prompt, image_reference_layout(3, 0, 0, 0, 0, 2, 2)}
        );
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected);
}
}  // namespace

int main() {
    batch_padding_keeps_requests_independent();
    using kadan::image::image_reference_layout;
    // Two text tokens, one 2x2 reference slot, three text tokens, then target.
    const auto layout = image_reference_layout(6, 2, 1, 2, 2, 2, 4);
    check(layout.prefix == 9);
    check(layout.positions.size() == 17);
    check(layout.segments.size() == 3);
    check(
        layout.segments[0].start == 0 && layout.segments[0].end == 2
        && layout.segments[0].text
    );
    check(
        layout.segments[1].start == 2 && layout.segments[1].end == 6
        && !layout.segments[1].text
    );
    check(
        layout.segments[2].start == 6 && layout.segments[2].end == 9
        && layout.segments[2].text
    );
    check((layout.positions[2] == std::array<std::int64_t, 3>{2, -1, -1}));
    check((layout.positions[5] == std::array<std::int64_t, 3>{2, 0, 0}));
    check((layout.positions[6] == std::array<std::int64_t, 3>{4, 4, 4}));
    check((layout.positions[9] == std::array<std::int64_t, 3>{7, -1, -2}));
    const auto text = image_reference_layout(3, 0, 0, 0, 0, 2, 2);
    check(text.prefix == 3 && text.segments.size() == 1);
    check((text.positions[3] == std::array<std::int64_t, 3>{3, -1, -1}));
    bool rejected = false;
    try {
        (void)image_reference_layout(6, 2, 2, 2, 2, 2, 4);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected);
}
