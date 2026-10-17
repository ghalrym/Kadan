#include "kadan/image_vae.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

#include "image_activation.hpp"

namespace kadan::image {
namespace {
void need(bool b, const char* e) {
    if (!b)
        throw std::runtime_error(e);
}

void stop(const std::atomic_bool& c) {
    need(!c.load(), "image_vae_cancelled");
}

void finite(std::span<const float> x) {
    for (float v : x)
        need(std::isfinite(v), "image_vae_nonfinite");
}

void disjoint(std::span<const float> a, std::span<const float> b) {
    const std::less<const float*> before;
    need(
        !before(a.data(), b.data() + b.size())
            || !before(b.data(), a.data() + a.size()),
        "image_decoder_input_overlap"
    );
}

void decoder_input_shape(
    std::span<const float> latent,
    std::span<const float> mean,
    std::span<const float> stddev,
    std::span<float> planar
) {
    need(
        !latent.empty() && latent.size() % 64 == 0 && latent.size() / 64 <= 18432
            && planar.size() == latent.size() && mean.size() == 64
            && stddev.size() == 64,
        "image_decoder_input_shape"
    );
    disjoint(latent, planar);
    disjoint(mean, planar);
    disjoint(stddev, planar);
    for (std::size_t c = 0; c < 64; ++c)
        need(
            std::isfinite(mean[c]) && std::isfinite(stddev[c]) && stddev[c] > 0,
            "image_latent_normalization"
        );
}

}  // namespace

void normalize_vae_features(
    std::span<const float> input,
    std::span<const float> gamma,
    std::size_t points,
    std::span<float> output,
    bool activate,
    const std::atomic_bool& cancel,
    DenseCompute* compute
) {
    stop(cancel);
    validate_vae_normalization(input, gamma, points, output);
    if (compute
        && compute->image_vae_normalize(
            input, gamma, points, output, activate, cancel
        )) {
        stop(cancel);
        return;
    }
    for (std::size_t point = 0; point < points; ++point) {
        if (point % 64 == 0)
            stop(cancel);
        double sum = 0;
        for (std::size_t c = 0; c < gamma.size(); ++c) {
            const double value = input[c * points + point];
            sum += value * value;
        }
        const auto scale = detail::vae_norm_scale(sum, gamma.size());
        for (std::size_t c = 0; c < gamma.size(); ++c)
            output[c * points + point] = detail::vae_norm_value(
                input[c * points + point], scale, gamma[c], activate
            );
    }
    stop(cancel);
    finite(output);
}

void prepare_decoder_input(
    std::span<const float> latent,
    std::span<const float> mean,
    std::span<const float> stddev,
    std::span<float> planar,
    const std::atomic_bool& cancel
) {
    stop(cancel);
    decoder_input_shape(latent, mean, stddev, planar);
    const auto points = latent.size() / 64;
    for (std::size_t p = 0; p < points; ++p) {
        stop(cancel);
        for (std::size_t c = 0; c < 64; ++c) {
            const float scaled = latent[p * 64 + c] * stddev[c];
            const float value = scaled + mean[c];
            need(std::isfinite(value), "image_vae_nonfinite");
            planar[c * points + p] = value;
        }
    }
    stop(cancel);
}

void prepare_decoder_input(
    MediaSession& session,
    std::span<float> latent_scratch,
    std::span<const float> mean,
    std::span<const float> stddev,
    std::span<float> planar,
    const std::atomic_bool& cancel
) {
    try {
        stop(cancel);
        decoder_input_shape(latent_scratch, mean, stddev, planar);
        disjoint(latent_scratch, mean);
        disjoint(latent_scratch, stddev);
        need(!session.plan().devices.empty(), "image_decoder_input_owner");
        const auto device = session.plan().devices.front().device;
        const auto owned = std::find_if(
            session.plan().tensors.begin(),
            session.plan().tensors.end(),
            [&](const auto& t) {
                return t.device == device && t.name == "ingress/latent";
            }
        );
        need(
            owned != session.plan().tensors.end() && owned->last == MediaPhase::complete
                && owned->bytes == latent_scratch.size_bytes(),
            "image_decoder_input_shape"
        );
        {
            auto latent = session.tensor(
                device,
                "ingress/latent",
                DeviceDType::f32,
                {latent_scratch.size() / 64, 64},
                {64, 1}
            );
            session.download_output(latent, std::as_writable_bytes(latent_scratch));
        }
        stop(cancel);
        prepare_decoder_input(latent_scratch, mean, stddev, planar, cancel);
    } catch (...) {
        session.abort(std::current_exception());
        throw;
    }
}

}  // namespace kadan::image
