#include "kadan/image_reference.hpp"

#include <source_location>
#include <stdexcept>

namespace {
void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error("image_reference_unit:" + std::to_string(at.line()));
}
}  // namespace

int main() {
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
