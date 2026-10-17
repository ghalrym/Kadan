#include "kadan/image_reference.hpp"

#include <cassert>

int main() {
    using kadan::image::image_reference_layout;
    // Two text tokens, one 2x2 reference slot, three text tokens, then target.
    const auto layout = image_reference_layout(6, 2, 1, 2, 2, 2, 4);
    assert(layout.prefix == 9);
    assert(layout.positions.size() == 17);
    assert(layout.segments.size() == 3);
    assert(
        layout.segments[0].start == 0 && layout.segments[0].end == 2
        && layout.segments[0].text
    );
    assert(
        layout.segments[1].start == 2 && layout.segments[1].end == 6
        && !layout.segments[1].text
    );
    assert(
        layout.segments[2].start == 6 && layout.segments[2].end == 9
        && layout.segments[2].text
    );
    assert((layout.positions[2] == std::array<std::int64_t, 3>{2, -1, -1}));
    assert((layout.positions[5] == std::array<std::int64_t, 3>{2, 0, 0}));
    assert((layout.positions[6] == std::array<std::int64_t, 3>{4, 4, 4}));
    assert((layout.positions[9] == std::array<std::int64_t, 3>{7, -1, -2}));
    const auto text = image_reference_layout(3, 0, 0, 0, 0, 2, 2);
    assert(text.prefix == 3 && text.segments.size() == 1);
    assert((text.positions[3] == std::array<std::int64_t, 3>{3, -1, -1}));
    bool rejected = false;
    try {
        (void)image_reference_layout(6, 2, 2, 2, 2, 2, 4);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}
