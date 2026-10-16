// Schedule, interpolation and compositing units only; no neural inference.
#include "kadan/image_edit.hpp"

#include <array>
#include <cassert>

int main() {
    using namespace kadan::image;
    const auto full = schedule(4096, 50);
    assert(edit_schedule(full, 0).steps == 0);
    assert(edit_schedule(full, 1).sigma == full.sigma);
    const auto partial = edit_schedule(full, .5f);
    assert(partial.steps == 25 && partial.sigma[0] == full.sigma[25]);
    assert(partial.sigma[25] == 0);
    assert(edit_schedule(full, .001f).steps == 1);
    Schedule turbo;
    turbo.steps = 8;
    turbo.sigma = {1, .9f, .8f, .6f, .4f, .3f, .2f, .1f, 0};
    assert(edit_schedule(turbo, .5f).sigma[0] == .4f);
    std::array<float, 8> pixels{1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<float, 8> original{-.75f, .5f, -.5f, 0, .25f, 1, -.25f, 0};
    composite_image_edit(pixels, original, std::array{0.f, .5f});
    for (std::size_t i = 0; i < 8; i += 2) {
        assert(pixels[i] == original[i]);
        assert(pixels[i + 1] == .5f + .5f * original[i + 1]);
    }
}
