#include "kadan/image_reference.hpp"

#include <array>
#include <source_location>
#include <stdexcept>

namespace {
void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error("image_reference_unit:" + std::to_string(at.line()));
}

template <typename F> void rejects(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    }
    check(false);
}
}  // namespace

int main() {
    using namespace kadan::image;
    const auto text_only = flux2_position_layout(3, 2, 2);
    check(text_only.text_tokens == 3);
    check(text_only.target_tokens == 4);
    check(text_only.reference_tokens == 0);
    check(text_only.positions.size() == 7);
    check((text_only.positions[0] == std::array<std::int64_t, 4>{0, 0, 0, 0}));
    check((text_only.positions[2] == std::array<std::int64_t, 4>{0, 0, 0, 2}));
    check((text_only.positions[3] == std::array<std::int64_t, 4>{0, 0, 0, 0}));
    check((text_only.positions[6] == std::array<std::int64_t, 4>{0, 1, 1, 0}));

    // Two ordered reference images receive distinct time offsets T=10 and T=20.
    const std::array<ImageGrid, 2> refs{ImageGrid{2, 3}, ImageGrid{2, 2}};
    const auto multi = flux2_position_layout(4, 2, 2, refs);
    check(multi.text_tokens == 4);
    check(multi.target_tokens == 4);
    check(multi.reference_tokens == 10);
    check(multi.positions.size() == 18);
    // Ref 0 starts at index 8 with T=10.
    check((multi.positions[8] == std::array<std::int64_t, 4>{10, 0, 0, 0}));
    check((multi.positions[13] == std::array<std::int64_t, 4>{10, 1, 2, 0}));
    // Ref 1 starts at index 14 with T=20.
    check((multi.positions[14] == std::array<std::int64_t, 4>{20, 0, 0, 0}));
    check((multi.positions[17] == std::array<std::int64_t, 4>{20, 1, 1, 0}));

    rejects([] { (void)flux2_position_layout(0, 2, 2); });
    rejects([] { (void)flux2_position_layout(4, 0, 2); });
    rejects([] {
        const std::array<ImageGrid, 5> too_many{
            ImageGrid{1, 1}, ImageGrid{1, 1}, ImageGrid{1, 1}, ImageGrid{1, 1}, ImageGrid{1, 1}
        };
        (void)flux2_position_layout(4, 2, 2, too_many);
    });
    rejects([] {
        const std::array<ImageGrid, 1> bad_ref{ImageGrid{0, 2}};
        (void)flux2_position_layout(4, 2, 2, bad_ref);
    });
}
