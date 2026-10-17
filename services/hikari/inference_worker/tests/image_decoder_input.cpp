// Scalar layout/normalization and fake transfer lifecycle only; no inference.
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <source_location>

#include "kadan/image_vae.hpp"

using namespace kadan;

namespace {
void check(bool ok, std::source_location at = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error(
            "image_decoder_input_test:" + std::to_string(at.line())
        );
}

template <class F> std::string fails(F fn) {
    try {
        fn();
    } catch (const std::exception& e) {
        return e.what();
    }
    throw std::runtime_error("image_decoder_input_expected_failure");
}

struct Inputs {
    std::array<float, 192> latent{}, planar{};
    std::array<float, 64> mean{}, stddev{};
    std::atomic_bool cancel = false;

    Inputs() {
        stddev.fill(1.f);
        planar.fill(-99.f);
        for (std::size_t i = 0; i < latent.size(); ++i)
            latent[i] = float(i) / 8.f - 10.f;
    }

    void prepare() {
        image::prepare_decoder_input(latent, mean, stddev, planar, cancel);
    }
};

void layout_and_rounding() {
    Inputs x;
    for (std::size_t c = 0; c < 64; ++c) {
        x.mean[c] = float(c) / 7.f;
        x.stddev[c] = float(c + 1) / 3.f;
    }
    // This product rounds before addition; an FMA would retain a nonzero result.
    x.latent[0] = 0x1.000002p0f;
    x.stddev[0] = 0x1.fffffcp-1f;
    x.mean[0] = -1.f;
    x.latent[1] = -0.f;
    x.mean[1] = -0.f;
    x.latent[2] = std::numeric_limits<float>::denorm_min();
    x.stddev[2] = 1.f;
    x.mean[2] = 0.f;
    const auto original = x.latent;
    x.prepare();
    for (std::size_t p = 0; p < 3; ++p)
        for (std::size_t c = 0; c < 64; ++c) {
            const float expected = x.latent[p * 64 + c] * x.stddev[c] + x.mean[c];
            check(
                std::bit_cast<std::uint32_t>(x.planar[c * 3 + p])
                == std::bit_cast<std::uint32_t>(expected)
            );
        }
    check(x.planar[0] == 0.f);
    check(std::fma(x.latent[0], x.stddev[0], x.mean[0]) != x.planar[0]);
    check(std::bit_cast<std::uint32_t>(x.planar[3]) == 0x80000000);
    check(std::bit_cast<std::uint32_t>(x.planar[6]) == 1);
    check(x.latent == original);
}

void invalid_host_inputs() {
    Inputs x;
    check(
        fails([&] {
            image::prepare_decoder_input(
                std::span(x.latent).first(191), x.mean, x.stddev, x.planar, x.cancel
            );
        })
        == "image_decoder_input_shape"
    );
    check(
        fails([&] {
            image::prepare_decoder_input(
                x.latent, x.mean, x.stddev, x.latent, x.cancel
            );
        })
        == "image_decoder_input_overlap"
    );
    check(
        fails([&] {
            image::prepare_decoder_input(
                x.latent, std::span(x.planar).first(64), x.stddev, x.planar, x.cancel
            );
        })
        == "image_decoder_input_overlap"
    );
    for (float invalid :
         {0.f,
          -1.f,
          std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()}) {
        x.stddev[63] = invalid;
        check(fails([&] { x.prepare(); }) == "image_latent_normalization");
    }
    x.stddev[63] = 1.f;
    x.mean[63] = std::numeric_limits<float>::infinity();
    check(fails([&] { x.prepare(); }) == "image_latent_normalization");
    for (float value : x.planar)
        check(value == -99.f);
}

void nonfinite_and_cancelled_host_input() {
    for (float invalid :
         {std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()}) {
        Inputs x;
        x.latent[0] = invalid;
        check(fails([&] { x.prepare(); }) == "image_vae_nonfinite");
    }
    Inputs x;
    x.latent[0] = std::numeric_limits<float>::max();
    x.stddev[0] = 2.f;
    check(fails([&] { x.prepare(); }) == "image_vae_nonfinite");
    x.cancel = true;
    check(fails([&] { x.prepare(); }) == "image_vae_cancelled");
    for (float value : x.planar)
        check(value == -99.f);
}

void normalization_cpu_formula() {
    std::array<float, 15> input{
        0.f,
        -0.f,
        1e-30f,
        1.f,
        3.f,
        0.f,
        -0.f,
        -1e-30f,
        -2.f,
        4.f,
        0.f,
        -0.f,
        2e-30f,
        3.f,
        -5.f
    };
    const std::array<float, 3> gamma{1.2345f, -3.14159f, .5f};
    std::array<float, 15> output{};
    std::atomic_bool cancel = false;
    for (bool activate : {false, true}) {
        image::normalize_vae_features(input, gamma, 5, output, activate, cancel);
        for (std::size_t p = 0; p < 5; ++p) {
            double sum = 0;
            for (std::size_t c = 0; c < 3; ++c)
                sum += double(input[c * 5 + p]) * input[c * 5 + p];
            const float scale = float(std::sqrt(3.) / std::max(std::sqrt(sum), 1e-12));
            for (std::size_t c = 0; c < 3; ++c) {
                const float value = input[c * 5 + p] * scale * gamma[c];
                const double v = value;
                const float expected =
                    activate ? float(
                                   v >= 0 ? v / (1 + std::exp(-v))
                                          : v * std::exp(v) / (1 + std::exp(v))
                               )
                             : value;
                check(
                    std::bit_cast<std::uint32_t>(output[c * 5 + p])
                    == std::bit_cast<std::uint32_t>(expected)
                );
            }
        }
    }
}

struct NormalizationDevice final : DenseCompute {
    bool supported = true, fail = false;
    unsigned calls = 0;
    std::atomic_bool* cancel_after = nullptr;

    void dense(
        std::span<const float>,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&,
        bool
    ) override {
        throw std::runtime_error("unexpected_projection");
    }

    bool image_vae_normalize(
        std::span<const float> input,
        std::span<const float> gamma,
        std::size_t points,
        std::span<float> output,
        bool activate,
        const std::atomic_bool& cancel
    ) override {
        ++calls;
        check(input.size() == 3 && gamma.size() == 1 && points == 3 && activate);
        check(!cancel.load());
        if (!supported)
            return false;
        output[0] = 17.f;  // Fake dispatch only, no device arithmetic.
        if (fail)
            throw std::runtime_error("normalization_device_failed");
        output[1] = 18.f;
        output[2] = 19.f;
        if (cancel_after)
            *cancel_after = true;
        return true;
    }
};

void normalization_dispatch_and_failures() {
    std::array<float, 3> input{1.f, -2.f, 3.f}, output{-99.f, -99.f, -99.f};
    std::array<float, 1> gamma{1.f};
    std::atomic_bool cancel = false;
    NormalizationDevice device;
    auto run = [&] {
        image::normalize_vae_features(input, gamma, 3, output, true, cancel, &device);
    };
    run();
    check(device.calls == 1 && output == std::array<float, 3>{17.f, 18.f, 19.f});
    device.supported = false;
    run();
    std::array<float, 3> reference{};
    image::normalize_vae_features(input, gamma, 3, reference, true, cancel);
    check(output == reference && device.calls == 2);
    device.supported = true;
    device.fail = true;
    check(fails(run) == "normalization_device_failed");
    check(output[0] == 17.f && output[1] == reference[1]);
    device.fail = false;
    device.cancel_after = &cancel;
    check(fails(run) == "image_vae_cancelled");
    const auto calls = device.calls;
    output.fill(-99.f);
    check(fails(run) == "image_vae_cancelled" && device.calls == calls);
    check(output == std::array<float, 3>{-99.f, -99.f, -99.f});
    cancel = false;
    for (auto invalid_points :
         {0ULL, 2ULL, std::numeric_limits<unsigned long long>::max()})
        check(
            fails([&] {
                image::normalize_vae_features(
                    input, gamma, invalid_points, output, false, cancel
                );
            })
            == "image_vae_norm_shape"
        );
    check(
        fails([&] {
            image::normalize_vae_features(input, gamma, 3, input, false, cancel);
        })
        == "image_vae_norm_overlap"
    );
    check(
        fails([&] {
            image::normalize_vae_features(
                input, std::span(output).first(1), 3, output, false, cancel
            );
        })
        == "image_vae_norm_overlap"
    );
    for (float invalid :
         {std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()}) {
        input[0] = invalid;
        check(
            fails([&] {
                image::normalize_vae_features(input, gamma, 3, output, false, cancel);
            })
            == "image_vae_nonfinite"
        );
        input[0] = 1.f;
        gamma[0] = invalid;
        check(
            fails([&] {
                image::normalize_vae_features(input, gamma, 3, output, true, cancel);
            })
            == "image_vae_nonfinite"
        );
        gamma[0] = 1.f;
    }
}

struct Trace {
    std::size_t allocations = 0, downloads = 0, synchronizations = 0;
    int downloaded_device = -1;
    bool fail_download = false;
    std::atomic_bool* cancel_after_download = nullptr;
};

struct Device final : MediaDeviceOperations {
    std::shared_ptr<Trace> trace;
    std::uint64_t event = 0;

    explicit Device(std::shared_ptr<Trace> t) : trace(std::move(t)) {
    }

    std::size_t allocation_alignment() const override {
        return 256;
    }

    void* stream(int) override {
        return nullptr;
    }

    Bytes free_memory(int device) override {
        return device == 0 ? 0 : 1000000;
    }

    bool direct_peer(int, int) override {
        return true;
    }

    void* allocate(int device, Bytes bytes, std::size_t) override {
        check(device == 1);
        ++trace->allocations;
        return ::operator new(bytes);
    }

    void free(int, void* p) override {
        --trace->allocations;
        ::operator delete(p);
    }

    void wait(int, MediaEvent) override {
    }

    MediaEvent record(int d) override {
        return {d, ++event};
    }

    void synchronize(int) override {
        ++trace->synchronizations;
    }

    void release_events(int) override {
    }

    void upload(int, void* dest, std::span<const std::byte> src) override {
        std::memcpy(dest, src.data(), src.size());
    }

    void download(int d, std::span<std::byte> dest, const void* src) override {
        ++trace->downloads;
        trace->downloaded_device = d;
        if (trace->fail_download)
            throw std::runtime_error("download_failed");
        std::memcpy(dest.data(), src, dest.size());
        if (trace->cancel_after_download)
            *trace->cancel_after_download = true;
    }

    void peer(int, const void*, int, void*, Bytes) override {
        throw std::runtime_error("unexpected_peer_copy");
    }

    std::uint32_t read_status(int, const void*) override {
        throw std::runtime_error("unexpected_kernel_status");
    }
};

struct Resident {
    Inputs input;
    std::array<float, 192> scratch{};
    std::shared_ptr<Trace> trace = std::make_shared<Trace>();
    std::shared_ptr<Resources> resources =
        std::make_shared<Resources>(Footprint{1000000, 1000000, 1000000});
    std::shared_ptr<MediaSessionActivity> activity =
        std::make_shared<MediaSessionActivity>();
    MediaSession session{
        resources,
        activity,
        std::make_unique<Device>(trace),
        Workload::image,
        {{0, 0, 0}, {1, 0, 0}},
        input.cancel
    };

    void admit(bool ready = true, bool complete = true, Bytes bytes = 192 * 4) {
        StageManifest manifest;
        // Only the named output boundary is modeled, never a denoiser/decoder.
        manifest.stage = MediaStage::image_decode;
        manifest.workspaces_complete = true;
        manifest.status_bytes = 4;
        manifest.tensors = {{"ingress/latent", bytes}};
        check(!session.admit(manifest, {3, 64, 64, 2, 32, 128, 0, 2}));
        check(session.plan().devices.front().device == 1);
        if (ready) {
            auto latent =
                session.tensor(1, "ingress/latent", DeviceDType::f32, {3, 64}, {64, 1});
            session.upload_input(latent, std::as_bytes(std::span(input.latent)));
        }
        if (complete)
            session.advance(MediaPhase::complete);
    }

    void prepare() {
        image::prepare_decoder_input(
            session, scratch, input.mean, input.stddev, input.planar, input.cancel
        );
    }

    void cleaned() {
        check(trace->allocations == 0 && activity->owners == 0);
        check(resources->snapshot().residents == 0);
    }
};

void resident_handoff() {
    Resident r;
    r.admit();
    const auto syncs = r.trace->synchronizations;
    r.prepare();
    check(r.scratch == r.input.latent);
    for (std::size_t p = 0; p < 3; ++p)
        for (std::size_t c = 0; c < 64; ++c)
            check(r.input.planar[c * 3 + p] == r.input.latent[p * 64 + c]);
    check(r.trace->downloads == 1 && r.trace->downloaded_device == 1);
    check(r.trace->synchronizations == syncs + 1);
    const auto& transfers = r.session.transfers();
    check(transfers.output_download == r.scratch.size() * sizeof(float));
    check(
        transfers.activation_download == 0 && transfers.activation_upload == 0
        && transfers.peer_activation == 0
    );
    check(r.activity->owners == 1);  // Caller retains ownership until finish.
    r.session.finish();
    r.cleaned();
}

void resident_requires_complete_certificate() {
    for (bool ready : {false, true}) {
        Resident r;
        r.admit(ready, !ready);
        check(fails([&] { r.prepare(); }) == "media_session_output_boundary");
        check(r.trace->downloads == 0);
        r.cleaned();
    }
    Resident r;
    r.admit(false, true, 128 * 4);
    check(fails([&] { r.prepare(); }) == "image_decoder_input_shape");
    check(r.trace->downloads == 0);
    r.cleaned();
}

void resident_cancellation_and_transfer_failure() {
    for (bool before : {false, true}) {
        Resident r;
        r.admit();
        if (before)
            r.input.cancel = true;
        else
            r.trace->cancel_after_download = &r.input.cancel;
        check(fails([&] { r.prepare(); }) == "image_vae_cancelled");
        check(r.trace->downloads == (before ? 0 : 1));
        for (float value : r.input.planar)
            check(value == -99.f);
        r.cleaned();
    }
    Resident r;
    r.admit();
    r.trace->fail_download = true;
    check(fails([&] { r.prepare(); }) == "download_failed");
    r.cleaned();
}
}  // namespace

int main() {
    layout_and_rounding();
    normalization_cpu_formula();
    normalization_dispatch_and_failures();
    invalid_host_inputs();
    nonfinite_and_cancelled_host_input();
    resident_handoff();
    resident_requires_complete_certificate();
    resident_cancellation_and_transfer_failure();
    std::cout << "image decoder input units passed (no inference)\n";
}
