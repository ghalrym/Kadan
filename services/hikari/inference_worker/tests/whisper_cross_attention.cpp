// Cache ownership/admission with fake projection fills only; no inference.
#include "kadan/whisper_cross_attention.hpp"

#include <limits>
#include <source_location>
#include <string_view>

namespace {
using namespace kadan;
using namespace kadan::stt;

void check(bool ok, std::source_location at = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error(
            "whisper_cross_attention_test:" + std::to_string(at.line())
        );
}

template <class F> std::string fails(F&& function) {
    try {
        function();
    } catch (const std::exception& error) {
        return error.what();
    }
    throw std::runtime_error("whisper_cross_attention_expected_failure");
}

void planning_and_admission() {
    check(WhisperCrossAttentionCache::bytes(24, 1500, 1024) == 294912000 + 512);
    check(WhisperCrossAttentionCache::bytes(32, 1500, 1280) == 491520000 + 512);
    for (auto dimension : {std::size_t(0), std::numeric_limits<std::size_t>::max()}) {
        check(
            fails([&] { WhisperCrossAttentionCache::bytes(dimension, 1, 1); })
            == "whisper_cross_attention_shape"
        );
        check(
            fails([&] { WhisperCrossAttentionCache::bytes(1, dimension, 1); })
            == "whisper_cross_attention_shape"
        );
        check(
            fails([&] { WhisperCrossAttentionCache::bytes(1, 1, dimension); })
            == "whisper_cross_attention_shape"
        );
    }
    const auto bytes = WhisperCrossAttentionCache::bytes(2, 3, 4);
    Resources resources({bytes, 50, 60});
    {
        WhisperCrossAttentionCache cache(resources, 2, 3, 4);
        check(resources.snapshot().used == Footprint({bytes, 0, 0}));
        check(resources.snapshot().residents == 1);
    }
    check(resources.snapshot().used == Footprint({0, 0, 0}));
    check(resources.snapshot().residents == 0);
    Resources insufficient({bytes - 1, 50, 60});
    check(
        fails([&] { WhisperCrossAttentionCache cache(insufficient, 2, 3, 4); })
        == "exhausted"
    );
    check(insufficient.snapshot().used == Footprint({0, 0, 0}));
    check(insufficient.snapshot().residents == 0);
    check(
        fails([&] {
            WhisperCrossAttentionCache cache(
                resources, 1, std::numeric_limits<std::size_t>::max(), 1280
            );
        })
        == "whisper_cross_attention_shape"
    );
    check(resources.snapshot().used == Footprint({0, 0, 0}));
    check(resources.snapshot().residents == 0);
}

void layer_reuse_and_request_scope() {
    const auto bytes = WhisperCrossAttentionCache::bytes(2, 3, 4);
    Resources resources({bytes});
    std::atomic_bool cancel{false};
    std::size_t calls = 0;
    auto fill = [&](std::span<float> keys, std::span<float> values) {
        ++calls;
        check(keys.size() == 12 && values.size() == 12);
        std::fill(keys.begin(), keys.end(), float(calls));
        std::fill(values.begin(), values.end(), -float(calls));
    };
    {
        WhisperCrossAttentionCache cache(resources, 2, 3, 4);
        const auto first = cache.get(0, cancel, fill);
        const auto repeat = cache.get(0, cancel, fill);
        check(calls == 1 && first.keys.data() == repeat.keys.data());
        check(first.keys.back() == 1 && first.values.back() == -1);
        const auto next = cache.get(1, cancel, fill);
        check(calls == 2 && next.keys.data() != first.keys.data());
        check(next.keys.front() == 2 && first.keys.front() == 1);
        check(
            fails([&] { cache.get(2, cancel, fill); })
            == "whisper_cross_attention_layer"
        );
        check(calls == 2);
        cancel = true;
        check(fails([&] { cache.get(0, cancel, fill); }) == "whisper_cancelled");
        check(calls == 2);
    }
    check(resources.snapshot().used[0] == 0);
    cancel = false;
    {
        WhisperCrossAttentionCache cache(resources, 2, 3, 4);
        const auto fresh = cache.get(0, cancel, fill);
        check(calls == 3 && fresh.keys.front() == 3 && fresh.values.front() == -3);
    }
    check(resources.snapshot().used[0] == 0 && resources.snapshot().residents == 0);
}

void failed_and_cancelled_fill() {
    const auto bytes = WhisperCrossAttentionCache::bytes(1, 3, 4);
    Resources resources({bytes});
    std::atomic_bool cancel{false};
    std::size_t calls = 0;
    {
        WhisperCrossAttentionCache cache(resources, 1, 3, 4);
        check(
            fails([&] {
                cache.get(0, cancel, [&](auto keys, auto) {
                    ++calls;
                    std::fill(keys.begin(), keys.end(), 8.f);
                    throw std::runtime_error("value_projection_failed");
                });
            })
            == "value_projection_failed"
        );
        check(
            fails([&] {
                cache.get(0, cancel, [&](auto keys, auto values) {
                    ++calls;
                    std::fill(keys.begin(), keys.end(), 1.f);
                    std::fill(values.begin(), values.end(), 1.f);
                    values.back() = std::numeric_limits<float>::infinity();
                });
            })
            == "whisper_nonfinite"
        );
        check(
            fails([&] {
                cache.get(0, cancel, [&](auto keys, auto values) {
                    ++calls;
                    std::fill(keys.begin(), keys.end(), 2.f);
                    std::fill(values.begin(), values.end(), 2.f);
                    cancel = true;
                });
            })
            == "whisper_cancelled"
        );
        cancel = false;
        const auto success = cache.get(0, cancel, [&](auto keys, auto values) {
            ++calls;
            std::fill(keys.begin(), keys.end(), 3.f);
            std::fill(values.begin(), values.end(), 4.f);
        });
        check(calls == 4 && success.keys.front() == 3 && success.values.back() == 4);
    }
    check(resources.snapshot().used[0] == 0 && resources.snapshot().residents == 0);
    check(
        fails([&] {
            WhisperCrossAttentionCache cache(resources, 1, 3, 4);
            cache.get(0, cancel, [&](auto, auto) {
                throw std::runtime_error("unwind_request");
            });
        })
        == "unwind_request"
    );
    check(resources.snapshot().used[0] == 0 && resources.snapshot().residents == 0);
}
}  // namespace

int main() {
    planning_and_admission();
    layer_reuse_and_request_scope();
    failed_and_cancelled_fill();
}
