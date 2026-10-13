// Checkpoint ingress/ownership with a fake device; no CUDA or projection execution.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <source_location>
#include <sstream>
#include <unistd.h>

#include "kadan/bf16_projection.hpp"
#include "kadan/image_denoiser.hpp"

void check(bool b, std::source_location at = std::source_location::current()) {
    if (!b)
        throw std::runtime_error("test_failed:" + std::to_string(at.line()));
}

// Checkpoint ingress and lifecycle metadata only; no projection arithmetic.
struct Device final : kadan::MediaDeviceOperations {
    kadan::Bytes available = 32 * 1024 * 1024, uploaded = 0;
    std::map<void*, kadan::Bytes> allocations;
    std::size_t probes = 0, next_event = 1;
    std::map<int, kadan::Bytes> remaining;
    bool metadata_execution = false;
    std::uint32_t status = 0;

    std::size_t allocation_alignment() const override {
        return 256;
    }

    void* stream(int) override {
        return nullptr;
    }

    kadan::Bytes free_memory(int d) override {
        ++probes;
        return remaining.try_emplace(d, available).first->second;
    }

    bool direct_peer(int, int) override {
        return metadata_execution;
    }

    void* allocate(int d, kadan::Bytes bytes, std::size_t alignment) override {
        check(probes >= 2);
        void* p = ::operator new(bytes, std::align_val_t(alignment));
        allocations.emplace(p, bytes);
        remaining.at(d) -= bytes;
        return p;
    }

    void free(int d, void* p) override {
        remaining.at(d) += allocations.at(p);
        allocations.erase(p);
        ::operator delete(p, std::align_val_t(256));
    }

    void wait(int, kadan::MediaEvent) override {
    }

    kadan::MediaEvent record(int d) override {
        return {d, next_event++};
    }

    void synchronize(int) override {
    }

    void release_events(int) override {
    }

    void upload(int, void*, std::span<const std::byte> raw) override {
        uploaded += raw.size();
        for (auto byte : raw)
            check(byte == std::byte{0});
    }

    void download(int, std::span<std::byte>, const void*) override {
        throw std::runtime_error("unexpected_download");
    }

    void peer(int, const void*, int, void*, kadan::Bytes) override {
        check(metadata_execution);
    }

    std::uint32_t read_status(int, const void*) override {
        check(metadata_execution);
        return status;
    }
};

struct WeightOptions {
    bool attention_f32 = false, denoiser = false, modulation_f32 = false,
         timestep_f32 = false, linear1_f32 = false;
    std::size_t layers = 1;
};

void write_weights(
    const char* directory,
    bool f32,
    std::size_t state = 2,
    std::size_t ff = 2,
    WeightOptions options = {}
) {
    std::ostringstream header;
    header << '{';
    std::size_t bytes = 0, count = 0;
    std::size_t block = options.denoiser ? 0 : 7;
    auto add = [&](std::string name, bool vector) {
        if (count++)
            header << ',';
        const auto rows =
            name == "img_mlp.proj" || name == "img_mlp.gate_layer" ? ff : state;
        const auto columns = name == "img_mlp.out" ? ff : state;
        const bool tensor_f32 =
            f32 || (options.attention_f32 && name == "attn.to_out.0");
        const auto size = (vector ? state : rows * columns) * (tensor_f32 ? 4 : 2);
        header << "\"transformer_blocks." << block << "." << name
               << ".weight\":{\"dtype\":\"" << (tensor_f32 ? "F32" : "BF16")
               << "\",\"shape\":[" << (vector ? state : rows);
        if (!vector)
            header << ',' << columns;
        header << "],\"data_offsets\":[" << bytes << ',' << bytes + size << "]}";
        bytes += size;
    };
    for (std::size_t layer = 0; layer < options.layers; ++layer) {
        block = options.denoiser ? layer : 7;
        for (auto name :
             {"attn.to_q",
              "attn.to_k",
              "attn.to_v",
              "attn.to_out.0",
              "img_mlp.proj",
              "img_mlp.gate_layer",
              "img_mlp.out"})
            add(name, false);
        for (auto name : {"attn.norm_q", "attn.norm_k"})
            add(name, true);
    }
    if (options.denoiser) {
        auto shared = [&](const char* name, std::size_t rows, std::size_t columns) {
            const bool fp32 =
                (options.modulation_f32 && std::string_view(name) == "modulation.1")
                || (options.timestep_f32
                    && std::string_view(name)
                           == "time_text_embed.timestep_embedder.linear_2")
                || (options.linear1_f32
                    && std::string_view(name)
                           == "time_text_embed.timestep_embedder.linear_1");
            const auto size = rows * (columns ? columns : 1) * (fp32 ? 4 : 2);
            header << ",\"" << name << ".weight\":{\"dtype\":\""
                   << (fp32 ? "F32" : "BF16") << "\",\"shape\":[" << rows;
            if (columns)
                header << ',' << columns;
            header << "],\"data_offsets\":[" << bytes << ',' << bytes + size << "]}";
            bytes += size;
        };
        shared("img_in", state, state);
        shared("txt_in.text_norm", state, 0);
        shared("txt_in.in_layer", state, state);
        shared("txt_in.out_layer", state, state);
        shared("time_text_embed.timestep_embedder.linear_1", state, 256);
        shared("time_text_embed.timestep_embedder.linear_2", state, state);
        shared("modulation.1", 4 * state, state);
        shared("norm_out.linear", state, state);
        shared("proj_out", state, state);
    }
    header << '}';
    const auto text = header.str();
    {
        std::ofstream file(
            std::string(directory) + "/weights.safetensors", std::ios::binary
        );
        for (unsigned i = 0; i < 8; ++i)
            file.put(char(std::uint64_t(text.size()) >> (8 * i)));
        file << text;
        for (std::size_t i = 0; i < bytes; ++i)
            file.put(0);
    }
}

// No-op launchers inspect metadata and submission order; never read tensor data.
struct Launcher final : kadan::video::ImageFeedForwardLauncher {
    Device& backend;
    int selected;
    bool fail_pair = false, fail_preparation = false, fail_attention = false,
         fail_projection = false, fail_modulation = false;
    bool fail_timestep = false, fail_timestep_row1 = false;
    bool fail_frequency = false, fail_linear1 = false, fail_linear1_row1 = false;
    std::atomic_bool* cancel_frequency = nullptr;
    std::atomic_bool* cancel_linear1 = nullptr;
    float expected_timestep = .625f;
    std::optional<float> supplied_timestep;
    void* frequencies = nullptr;
    std::uint64_t frequency_generation = 0;
    void* linear1 = nullptr;
    std::uint64_t linear1_generation = 0;
    std::vector<std::size_t> linear1_rows;
    std::atomic_bool* cancel_timestep = nullptr;
    void* activated = nullptr;
    std::uint64_t activated_generation = 0;
    std::vector<std::size_t> timestep_rows;
    void* modulation = nullptr;
    std::uint64_t modulation_generation = 0;
    std::vector<std::size_t> time_rows;
    void* attended = nullptr;
    std::uint64_t attended_generation = 0;
    std::vector<std::size_t> projection_first;
    std::vector<char> calls;
    std::vector<std::size_t> residual_first;

    Launcher(Device& d, int id) : backend(d), selected(id) {
    }

    int device() const override {
        return selected;
    }

    void frequency(
        void*, std::span<const kadan::MediaDeviceSpan> write, float timestep
    ) override {
        check(write.size() == 2 && calls.empty());
        kadan::video::validate_image_frequency(selected, write[0], write[1], timestep);
        check(timestep == expected_timestep);
        supplied_timestep = timestep;
        frequencies = write[0].data;
        frequency_generation = write[0].generation;
        calls.push_back('f');
        backend.status = fail_frequency ? 1 : 0;
        if (cancel_frequency)
            cancel_frequency->store(true);
    }

    void timestep(
        void*,
        std::span<const kadan::MediaDeviceSpan> read,
        std::span<const kadan::MediaDeviceSpan> write
    ) override {
        check(read.size() == 2 && write.size() == 4);
        const kadan::MediaDeviceSpan bias{
            nullptr, selected, kadan::DeviceDType::f32, {}, {}, 0, 0, 0
        };
        kadan::video::validate_bf16_projection(
            selected, read[0], read[1], bias, write[1], write[0], write[2], write[3]
        );
        if (read[0].dimensions[1] == 256) {
            check(read[0].dimensions == std::vector<std::size_t>{1, 256});
            check(read[1].dimensions == std::vector<std::size_t>{64, 256});
            check(write[0].dimensions == std::vector<std::size_t>{1, 64});
            check(
                calls == std::vector<char>{'f'} || calls == std::vector<char>{'f', 'l'}
            );
            check(
                static_cast<std::byte*>(read[0].data) - read[0].offset == frequencies
                && read[0].generation == frequency_generation
            );
            linear1 = static_cast<std::byte*>(write[0].data) - write[0].offset;
            linear1_generation = write[0].generation;
            linear1_rows.push_back(read[0].offset / 1024);
            calls.push_back('l');
            backend.status =
                fail_linear1 || (fail_linear1_row1 && linear1_rows.size() == 2) ? 1 : 0;
            if (cancel_linear1)
                cancel_linear1->store(true);
            return;
        }
        check(read[0].dimensions == std::vector<std::size_t>{1, 64});
        check(read[1].dimensions == std::vector<std::size_t>{64, 64});
        check(write[0].dimensions == std::vector<std::size_t>{1, 64});
        check(
            calls == std::vector<char>{'f', 'l', 'l'}
            || calls == std::vector<char>{'f', 'l', 'l', 't'}
        );
        check(linear1_rows == std::vector<std::size_t>{0, 1});
        check(
            static_cast<std::byte*>(read[0].data) - read[0].offset == linear1
            && read[0].generation == linear1_generation
        );
        activated = static_cast<std::byte*>(write[0].data) - write[0].offset;
        activated_generation = write[0].generation;
        timestep_rows.push_back(read[0].offset / 256);
        calls.push_back('t');
        backend.status =
            fail_timestep || (fail_timestep_row1 && timestep_rows.size() == 2) ? 1 : 0;
        if (cancel_timestep)
            cancel_timestep->store(true);
    }

    void standard(
        void*,
        std::span<const kadan::MediaDeviceSpan> read,
        std::span<const kadan::MediaDeviceSpan> write
    ) override {
        check(read.size() == 2 && write.size() == 4);
        const kadan::MediaDeviceSpan bias{
            nullptr, selected, kadan::DeviceDType::f32, {}, {}, 0, 0, 0
        };
        kadan::video::validate_bf16_projection(
            selected, read[0], read[1], bias, write[1], write[0], write[2], write[3]
        );
        if (read[1].dimensions[0] == 256) {
            check(read[0].dimensions == std::vector<std::size_t>{1, 64});
            check(write[0].dimensions == std::vector<std::size_t>{1, 256});
            check(std::all_of(calls.begin(), calls.end(), [](char c) {
                return c == 'f' || c == 'l' || c == 't' || c == 'm';
            }));
            check(timestep_rows == std::vector<std::size_t>{0, 1});
            check(
                static_cast<std::byte*>(read[0].data) - read[0].offset == activated
                && read[0].generation == activated_generation
            );
            modulation = static_cast<std::byte*>(write[0].data) - write[0].offset;
            modulation_generation = write[0].generation;
            time_rows.push_back(read[0].offset / 256);
            calls.push_back('m');
            backend.status = fail_modulation ? 1 : 0;
            return;
        }
        check(std::all_of(calls.begin(), calls.end(), [](char c) {
            return c == 'o' || c == 'f' || c == 'l' || c == 't' || c == 'm';
        }));
        attended = static_cast<std::byte*>(read[0].data) - read[0].offset;
        attended_generation = read[0].generation;
        projection_first.push_back(read[0].offset / (read[0].dimensions[1] * 4));
        calls.push_back('o');
        backend.status = fail_projection ? 1 : 0;
    }

    void attention_residual(
        void*,
        std::span<const kadan::MediaDeviceSpan> read,
        std::span<const kadan::MediaDeviceSpan> write,
        std::size_t text
    ) override {
        check(read.size() == 3 && write.size() == 3 && text == 3);
        kadan::video::validate_image_attention_residual(
            selected, read[0], read[1], read[2], write[0], write[1], write[2], text
        );
        check(!calls.empty() && calls.back() == 'o');
        check(read[1].data == attended && read[1].generation == attended_generation);
        check(read[1].dimensions == std::vector<std::size_t>{17, 64});
        if (modulation)
            check(
                read[2].data == modulation
                && read[2].generation == modulation_generation
            );
        calls.push_back('a');
        backend.status = fail_attention ? 1 : 0;
    }

    void prepare(
        void*,
        std::span<const kadan::MediaDeviceSpan> read,
        std::span<const kadan::MediaDeviceSpan> write,
        std::size_t text
    ) override {
        check(read.size() == 2 && write.size() == 3 && text == 3);
        kadan::video::validate_image_modulation(
            selected, read[0], read[1], write[0], write[1], write[2], text
        );
        check(calls.empty() || calls.back() == 'a' || calls.back() == 'm');
        if (modulation)
            check(
                read[1].data == modulation
                && read[1].generation == modulation_generation
            );
        calls.push_back('n');
        backend.status = fail_preparation ? 1 : 0;
    }

    void pair(
        void*,
        std::span<const kadan::MediaDeviceSpan> read,
        std::span<const kadan::MediaDeviceSpan> write
    ) override {
        check(read.size() == 3 && write.size() == 6);
        const auto halves =
            kadan::video::split_image_silu_pair(write[0], read[0].dimensions[0]);
        kadan::video::validate_image_silu_pair(
            selected,
            read[0],
            read[1],
            read[2],
            write[1],
            halves[0],
            halves[1],
            write[2],
            write[3],
            write[4],
            write[5]
        );
        calls.push_back('p');
        backend.status = fail_pair ? 1 : 0;
    }

    void down(
        void*,
        std::span<const kadan::MediaDeviceSpan> read,
        std::span<const kadan::MediaDeviceSpan> write
    ) override {
        check(read.size() == 2 && write.size() == 4 && calls.back() == 'p');
        const kadan::MediaDeviceSpan bias{
            nullptr, selected, kadan::DeviceDType::f32, {}, {}, 0, 0, 0
        };
        kadan::video::validate_bf16_projection(
            selected, read[0], read[1], bias, write[1], write[0], write[2], write[3]
        );
        calls.push_back('d');
    }

    void residual(
        void*,
        const kadan::MediaDeviceSpan& hidden,
        const kadan::MediaDeviceSpan& projected,
        const kadan::MediaDeviceSpan& gates,
        const kadan::MediaDeviceSpan& status,
        std::size_t first,
        std::size_t text
    ) override {
        check(calls.back() == 'd' && text == 3);
        kadan::video::validate_image_residual(
            selected, hidden, projected, gates, status, first, text
        );
        residual_first.push_back(first);
        calls.push_back('r');
    }
};

int main() {
    char directory[] = "/tmp/kadan-weight-load-XXXXXX";
    check(mkdtemp(directory));

    struct Cleanup {
        const char* path;

        ~Cleanup() {
            std::filesystem::remove_all(path);
        }
    } cleanup{directory};

    write_weights(directory, false);
    auto resources = std::make_shared<kadan::Resources>(
        kadan::Footprint{64 * 1024 * 1024, 32 * 1024 * 1024}
    );
    kadan::image::TransformerBlock block(resources);
    std::atomic_bool cancel = false;
    block.load(directory, "weights.safetensors", 7, {2, 1, 2, 2, {0, 0, 2}}, cancel);
    check(resources->snapshot().used[0] > 0);
    auto backend = std::make_unique<Device>();
    auto& device = *backend;
    auto activity = std::make_shared<kadan::MediaSessionActivity>();
    kadan::MediaSession session(
        resources, activity, std::move(backend), kadan::Workload::image, {{0}}, cancel
    );
    check(!block.prepare_feed_forward(
        session, 5, kadan::image::HiddenBoundary::post_attention, cancel
    ));
    const auto& plan = session.plan();
    check(plan.weights.size() == 3 && device.uploaded == 24);
    for (const auto& slice : plan.weights) {
        check(slice.name.starts_with("transformer_blocks.7.img_mlp."));
        check(slice.stored_bytes == 8 && slice.identity.inode != 0);
        check(slice.first_row == 0 && slice.rows == 2);
    }
    check(session.evidence().budgets[0].physical_free == 32 * 1024 * 1024);
    auto normalized = session.tensor(
        0, "resident/normalized", kadan::DeviceDType::f32, {5, 2}, {2, 1}
    );
    check(normalized.layout().bytes == 40);
    normalized = {};
    session.advance(kadan::MediaPhase::ff);
    auto aggregate =
        session.tensor(0, "resident/status", kadan::DeviceDType::u32, {1}, {1});
    auto gate = session.tensor(0, "ff/gate-status", kadan::DeviceDType::u32, {1}, {1});
    auto up = session.tensor(0, "ff/up-status", kadan::DeviceDType::u32, {1}, {1});
    check(aggregate.layout().data != gate.layout().data);
    check(up.layout().data != gate.layout().data);
    check(up.layout().data != aggregate.layout().data);
    aggregate = {};
    gate = {};
    up = {};
    session.finish();
    check(resources->snapshot().used[1] == 0 && device.allocations.empty());
    block.unload();
    check(resources->snapshot().used[0] == 0);
    write_weights(directory, false, 64, 128);
    constexpr kadan::Bytes budget = 8 * 1024 * 1024 + 54692;
    auto tiled_resources = std::make_shared<kadan::Resources>(
        kadan::Footprint{64 * 1024 * 1024, budget, budget}
    );

    struct Compute final : kadan::DenseCompute {
        bool supports_bf16_weights() const override {
            return true;
        }

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
            throw std::runtime_error("unexpected_dense");
        }
    };

    kadan::image::TransformerBlock tiled_block(
        tiled_resources, std::make_shared<Compute>()
    );
    tiled_block.load(
        directory, "weights.safetensors", 7, {64, 1, 64, 128, {0, 0, 64}}, cancel
    );
    enum class Scenario {
        missing_boundary,
        pair_failure,
        tiled
    };
    for (const auto scenario :
         {Scenario::missing_boundary, Scenario::pair_failure, Scenario::tiled}) {
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        metadata.available = budget;
        metadata.metadata_execution = true;
        kadan::MediaSession resident(
            tiled_resources,
            activity,
            std::move(fake),
            kadan::Workload::image,
            {{0}, {1}},
            cancel
        );
        check(!tiled_block.prepare_feed_forward(
            resident, 17, kadan::image::HiddenBoundary::post_attention, cancel
        ));
        check(resident.plan().devices.size() == 2 && resident.plan().rows < 17);
        for (const auto& d : resident.plan().devices) {
            for (auto name :
                 {"resident/normalized", "resident/hidden", "ff/residual-gates"}) {
                const std::size_t rows =
                    std::string_view(name) == "ff/residual-gates" ? 2 : 17;
                auto view = resident.tensor(
                    d.device, name, kadan::DeviceDType::f32, {rows, 64}, {64, 1}
                );
                if (scenario != Scenario::missing_boundary
                    || std::string_view(name) != "resident/normalized") {
                    std::vector<std::byte> ingress(view.layout().bytes);
                    resident.upload_input(view, ingress);
                }
            }
        }
        Launcher first(metadata, 0), second(metadata, 1);
        first.fail_pair = scenario == Scenario::pair_failure;
        const std::array<kadan::video::ImageFeedForwardLauncher*, 2> launchers{
            &first, &second
        };
        bool rejected = false;
        try {
            tiled_block.execute_feed_forward(resident, 17, 3, launchers, cancel);
        } catch (const std::exception& e) {
            rejected = true;
            check(
                std::string_view(e.what())
                == (scenario == Scenario::missing_boundary
                        ? "media_session_unready_input"
                        : "media_session_nonfinite_output")
            );
        }
        check(rejected == (scenario != Scenario::tiled));
        if (scenario == Scenario::missing_boundary)
            check(first.calls.empty() && second.calls.empty());
        if (scenario == Scenario::pair_failure)
            check(first.calls == std::vector<char>{'p'} && second.calls.empty());
        if (scenario == Scenario::tiled) {
            check(
                first.calls.size() == 3 * first.residual_first.size()
                && first.calls == second.calls
            );
            check(first.residual_first.size() > 1 && first.residual_first.front() == 0);
            check(first.residual_first.back() + resident.plan().rows >= 17);
            check(
                resident.transfers().peer_activation
                && resident.transfers().local_activation
            );
            check(
                !resident.transfers().activation_upload
                && !resident.transfers().activation_download
                && !resident.transfers().output_download
            );
        }
        if (!rejected)
            resident.finish();
        check(
            metadata.allocations.empty() && tiled_resources->snapshot().used[1] == 0
            && tiled_resources->snapshot().used[2] == 0
        );
    }
    enum class PreparationScenario {
        complete,
        missing_hidden,
        missing_modulation,
        missing_attended,
        failed_projection,
        failed_attention,
        failed_preparation
    };
    using kadan::image::HiddenBoundary;
    auto preparation_case = [&](HiddenBoundary boundary,
                                PreparationScenario scenario,
                                const std::vector<char>& expected_first,
                                const std::vector<char>& expected_second) {
        const bool attention = boundary == HiddenBoundary::pre_attention;
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        metadata.available = budget;
        metadata.metadata_execution = true;
        kadan::MediaSession resident(
            tiled_resources,
            activity,
            std::move(fake),
            kadan::Workload::image,
            {{0}, {1}},
            cancel
        );
        check(!tiled_block.prepare_feed_forward(resident, 17, boundary, cancel));
        check(resident.plan().devices.size() == 2 && resident.plan().rows < 17);
        if (attention)
            check(
                std::count_if(
                    resident.plan().weights.begin(),
                    resident.plan().weights.end(),
                    [](const auto& w) {
                        return w.name == "transformer_blocks.7.attn.to_out.0.weight"
                               && w.stored_bytes == w.rows * 64 * 2;
                    }
                )
                == 2
            );
        for (const auto& d : resident.plan().devices) {
            auto hidden = resident.tensor(
                d.device, "resident/hidden", kadan::DeviceDType::f32, {17, 64}, {64, 1}
            );
            auto modulation = resident.tensor(
                d.device, "ff/modulation", kadan::DeviceDType::f32, {2, 256}, {256, 1}
            );
            if (d.device != 1 || scenario != PreparationScenario::missing_hidden) {
                std::vector<std::byte> ingress(hidden.layout().bytes);
                resident.upload_input(hidden, ingress);
            }
            if (d.device != 1 || scenario != PreparationScenario::missing_modulation) {
                std::vector<std::byte> ingress(modulation.layout().bytes);
                resident.upload_input(modulation, ingress);
            }
            if (attention
                && (d.device != 1
                    || scenario != PreparationScenario::missing_attended)) {
                auto attended = resident.tensor(
                    d.device,
                    "resident/normalized",
                    kadan::DeviceDType::f32,
                    {17, 64},
                    {64, 1}
                );
                std::vector<std::byte> ingress(attended.layout().bytes);
                resident.upload_input(attended, ingress);
            }
        }
        Launcher first(metadata, 0), second(metadata, 1);
        first.fail_projection = scenario == PreparationScenario::failed_projection;
        first.fail_preparation = scenario == PreparationScenario::failed_preparation;
        first.fail_attention = scenario == PreparationScenario::failed_attention;
        const std::array<kadan::video::ImageFeedForwardLauncher*, 2> launchers{
            &first, &second
        };
        bool rejected = false;
        try {
            tiled_block.prepare_feed_forward_inputs(
                resident, 17, 3, boundary, launchers, cancel
            );
            tiled_block.execute_feed_forward(resident, 17, 3, launchers, cancel);
        } catch (const std::exception& e) {
            rejected = true;
            check(
                std::string_view(e.what())
                == ((scenario == PreparationScenario::failed_projection
                     || scenario == PreparationScenario::failed_attention
                     || scenario == PreparationScenario::failed_preparation)
                        ? "media_session_nonfinite_output"
                        : "media_session_unready_input")
            );
        }
        check(rejected == (scenario != PreparationScenario::complete));
        const auto projection_tiles =
            (17 + resident.plan().rows - 1) / resident.plan().rows;
        auto with_projection = [&](const std::vector<char>& tail) {
            if (!attention || scenario == PreparationScenario::missing_hidden
                || scenario == PreparationScenario::missing_modulation
                || scenario == PreparationScenario::missing_attended)
                return tail;
            std::vector<char> result(projection_tiles, 'o');
            result.insert(result.end(), tail.begin(), tail.end());
            return result;
        };
        const auto first_expected = with_projection(expected_first);
        const auto second_expected = with_projection(expected_second);
        if (scenario == PreparationScenario::failed_projection) {
            check(first.calls == std::vector<char>{'o'} && second.calls.empty());
        } else if (scenario == PreparationScenario::complete) {
            check(first.calls == second.calls);
            check(
                first.calls.size()
                == first_expected.size() + 3 * first.residual_first.size()
            );
            check(
                std::equal(
                    first_expected.begin(), first_expected.end(), first.calls.begin()
                )
            );
            check(first.residual_first.size() > 1);
            if (attention) {
                check(first.projection_first == second.projection_first);
                check(first.projection_first.size() == projection_tiles);
                check(first.projection_first.front() == 0);
                check(first.projection_first.back() + resident.plan().rows >= 17);
            }
            check(
                !resident.transfers().activation_upload
                && !resident.transfers().activation_download
            );
            resident.finish();
        } else {
            check(first.calls == first_expected && second.calls == second_expected);
        }
        check(
            metadata.allocations.empty() && tiled_resources->snapshot().used[1] == 0
            && tiled_resources->snapshot().used[2] == 0
        );
    };
    preparation_case(
        HiddenBoundary::post_attention, PreparationScenario::complete, {'n'}, {'n'}
    );
    preparation_case(
        HiddenBoundary::post_attention, PreparationScenario::missing_hidden, {}, {}
    );
    preparation_case(
        HiddenBoundary::post_attention, PreparationScenario::missing_modulation, {}, {}
    );
    preparation_case(
        HiddenBoundary::post_attention,
        PreparationScenario::failed_preparation,
        {'n'},
        {}
    );

    preparation_case(
        HiddenBoundary::pre_attention,
        PreparationScenario::complete,
        {'a', 'n'},
        {'a', 'n'}
    );
    preparation_case(
        HiddenBoundary::pre_attention, PreparationScenario::missing_hidden, {}, {}
    );
    preparation_case(
        HiddenBoundary::pre_attention, PreparationScenario::missing_modulation, {}, {}
    );
    preparation_case(
        HiddenBoundary::pre_attention, PreparationScenario::missing_attended, {}, {}
    );
    preparation_case(
        HiddenBoundary::pre_attention, PreparationScenario::failed_projection, {}, {}
    );
    preparation_case(
        HiddenBoundary::pre_attention, PreparationScenario::failed_attention, {'a'}, {}
    );
    preparation_case(
        HiddenBoundary::pre_attention,
        PreparationScenario::failed_preparation,
        {'a', 'n'},
        {'a'}
    );
    tiled_block.unload();
    write_weights(directory, false, 64, 128, {.denoiser = true});
    auto conditioning_resources = std::make_shared<kadan::Resources>(
        kadan::Footprint{128 * 1024 * 1024, budget + 78592, budget + 78592}
    );
    kadan::image::Denoiser denoiser(
        conditioning_resources, std::make_shared<Compute>()
    );
    const std::array<std::string, 1> files{"weights.safetensors"};
    const kadan::image::DenoiserConfig config{{64, 1, 64, 128, {0, 0, 64}}, 1, 64, 64};
    denoiser.load(directory, files, config, cancel);
    enum class ConditioningScenario {
        complete,
        failed_frequency,
        failed_second_frequency,
        cancelled_frequency,
        failed_linear1,
        failed_second_linear1,
        failed_linear1_row1,
        cancelled_linear1,
        invalid_negative,
        invalid_overshoot,
        invalid_nan,
        invalid_infinity,
        missing_hidden,
        missing_attended,
        failed_modulation,
        failed_timestep,
        failed_second_timestep,
        failed_timestep_row1,
        supplied_output,
        cancelled_timestep,
        wrong_boundary,
        missing_launcher,
        missing_conditioning
    };
    auto conditioning_case = [&](HiddenBoundary boundary,
                                 ConditioningScenario scenario) {
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        metadata.available = budget + 78592;
        metadata.metadata_execution = true;
        kadan::MediaSession resident(
            conditioning_resources,
            activity,
            std::move(fake),
            kadan::Workload::image,
            {{0}, {1}},
            cancel
        );
        check(!denoiser.prepare_feed_forward(resident, 0, 17, boundary, cancel));
        check(resident.plan().devices.size() == 2 && resident.plan().rows < 17);
        check(
            std::count_if(
                resident.plan().weights.begin(),
                resident.plan().weights.end(),
                [](const auto& w) {
                    return w.name == "modulation.1.weight" && w.first_row == 0
                           && w.rows == 256 && w.stored_bytes == 32768
                           && w.identity.inode != 0;
                }
            )
            == 2
        );
        check(
            std::count_if(
                resident.plan().weights.begin(),
                resident.plan().weights.end(),
                [](const auto& w) {
                    return w.name == "time_text_embed.timestep_embedder.linear_2.weight"
                           && w.first_row == 0 && w.rows == 64 && w.stored_bytes == 8192
                           && w.identity.inode != 0;
                }
            )
            == 2
        );
        check(
            std::count_if(
                resident.plan().weights.begin(),
                resident.plan().weights.end(),
                [](const auto& w) {
                    return w.name == "time_text_embed.timestep_embedder.linear_1.weight"
                           && w.first_row == 0 && w.rows == 64
                           && w.stored_bytes == 32768 && w.identity.inode != 0;
                }
            )
            == 2
        );
        for (const auto& d : resident.plan().devices) {
            for (const auto* name : {"resident/hidden", "resident/normalized"}) {
                const bool hidden = std::string_view(name) == "resident/hidden";
                if ((d.device == 1
                     && ((hidden && scenario == ConditioningScenario::missing_hidden)
                         || (!hidden
                             && scenario == ConditioningScenario::missing_attended)))
                    || (!hidden && boundary == HiddenBoundary::post_attention))
                    continue;
                auto view = resident.tensor(
                    d.device, name, kadan::DeviceDType::f32, {17, 64}, {64, 1}
                );
                std::vector<std::byte> ingress(view.layout().bytes);
                resident.upload_input(view, ingress);
            }
            if (d.device == 1 && scenario == ConditioningScenario::supplied_output) {
                for (const char* name :
                     {"conditioning/frequency",
                      "conditioning/time-linear1",
                      "conditioning/time-activated"}) {
                    const std::size_t columns =
                        std::string_view(name) == "conditioning/frequency" ? 256 : 64;
                    auto output = resident.tensor(
                        d.device,
                        name,
                        kadan::DeviceDType::f32,
                        {2, columns},
                        {columns, 1}
                    );
                    std::vector<std::byte> ingress(output.layout().bytes);
                    resident.upload_input(output, ingress);
                }
            }
        }
        Launcher first(metadata, 0), second(metadata, 1);
        first.fail_frequency = scenario == ConditioningScenario::failed_frequency;
        second.fail_frequency =
            scenario == ConditioningScenario::failed_second_frequency;
        first.cancel_frequency =
            scenario == ConditioningScenario::cancelled_frequency ? &cancel : nullptr;
        first.fail_linear1 = scenario == ConditioningScenario::failed_linear1;
        second.fail_linear1 = scenario == ConditioningScenario::failed_second_linear1;
        first.fail_linear1_row1 = scenario == ConditioningScenario::failed_linear1_row1;
        first.cancel_linear1 =
            scenario == ConditioningScenario::cancelled_linear1 ? &cancel : nullptr;
        first.fail_timestep_row1 =
            scenario == ConditioningScenario::failed_timestep_row1;
        first.fail_modulation = scenario == ConditioningScenario::failed_modulation;
        first.fail_timestep = scenario == ConditioningScenario::failed_timestep;
        second.fail_timestep = scenario == ConditioningScenario::failed_second_timestep;
        first.cancel_timestep =
            scenario == ConditioningScenario::cancelled_timestep ? &cancel : nullptr;
        const std::array<kadan::video::ImageFeedForwardLauncher*, 2> launchers{
            &first,
            scenario == ConditioningScenario::missing_launcher ? &first : &second
        };
        std::optional<float> timestep = .625f;
        if (scenario == ConditioningScenario::missing_conditioning)
            timestep.reset();
        if (scenario == ConditioningScenario::invalid_negative)
            timestep = -.01f;
        if (scenario == ConditioningScenario::invalid_overshoot)
            timestep = 1.01f;
        if (scenario == ConditioningScenario::invalid_nan)
            timestep = std::numeric_limits<float>::quiet_NaN();
        if (scenario == ConditioningScenario::invalid_infinity)
            timestep = std::numeric_limits<float>::infinity();
        bool rejected = false;
        try {
            denoiser.execute_feed_forward(
                resident,
                0,
                17,
                3,
                timestep,
                scenario == ConditioningScenario::wrong_boundary
                    ? HiddenBoundary::pre_attention
                    : boundary,
                launchers,
                cancel
            );
        } catch (const std::exception& e) {
            rejected = true;
            const std::string_view error(e.what());
            check(
                error == "media_session_nonfinite_output"
                || error == "media_session_unready_input"
                || error == "image_ff_launcher" || error == "image_ff_plan"
                || error == "image_cancelled" || error == "media_session_cancelled"
                || error == "image_frequency_layout"
            );
        }
        cancel.store(false);
        const bool complete = scenario == ConditioningScenario::complete
                              || scenario == ConditioningScenario::supplied_output;
        check(rejected != complete);
        if (complete) {
            check(
                first.calls == second.calls && first.calls[0] == 'f'
                && first.calls[1] == 'l' && first.calls[2] == 'l'
                && first.calls[3] == 't' && first.calls[4] == 't'
                && first.calls[5] == 'm' && first.calls[6] == 'm'
            );
            check(
                first.linear1_rows == std::vector<std::size_t>{0, 1}
                && first.linear1_rows == second.linear1_rows
            );
            check(
                first.timestep_rows == std::vector<std::size_t>{0, 1}
                && first.timestep_rows == second.timestep_rows
            );
            check(
                first.time_rows == std::vector<std::size_t>{0, 1}
                && first.time_rows == second.time_rows
            );
            check(std::count(first.calls.begin(), first.calls.end(), 'n') == 1);
            check(
                std::count(first.calls.begin(), first.calls.end(), 'a')
                == (boundary == HiddenBoundary::pre_attention ? 1 : 0)
            );
            check(first.residual_first.size() > 1);
            auto hidden = resident.tensor(
                1, "resident/hidden", kadan::DeviceDType::f32, {17, 64}, {64, 1}
            );
            const std::array ready{&hidden};
            resident.require_ready(ready);
            hidden = {};
            check(
                !resident.transfers().activation_upload
                && !resident.transfers().activation_download
                && !resident.transfers().output_download
            );
            resident.finish();
        } else if (scenario == ConditioningScenario::failed_modulation) {
            check(
                first.calls == std::vector<char>{'f', 'l', 'l', 't', 't', 'm'}
                && second.calls.empty()
            );
        } else if (
            scenario == ConditioningScenario::failed_timestep
            || scenario == ConditioningScenario::cancelled_timestep
        ) {
            check(
                first.calls == std::vector<char>{'f', 'l', 'l', 't'}
                && second.calls.empty()
            );
        } else if (scenario == ConditioningScenario::failed_timestep_row1) {
            check(
                first.calls == std::vector<char>{'f', 'l', 'l', 't', 't'}
                && second.calls.empty()
            );
        } else if (scenario == ConditioningScenario::failed_second_timestep) {
            check(
                first.calls == std::vector<char>{'f', 'l', 'l', 't', 't', 'm', 'm'}
                && second.calls == std::vector<char>{'f', 'l', 'l', 't'}
            );
        } else if (
            scenario == ConditioningScenario::failed_frequency
            || scenario == ConditioningScenario::cancelled_frequency
        ) {
            check(first.calls == std::vector<char>{'f'} && second.calls.empty());
        } else if (
            scenario == ConditioningScenario::failed_linear1
            || scenario == ConditioningScenario::cancelled_linear1
        ) {
            check(first.calls == std::vector<char>{'f', 'l'} && second.calls.empty());
        } else if (scenario == ConditioningScenario::failed_linear1_row1) {
            check(
                first.calls == std::vector<char>{'f', 'l', 'l'} && second.calls.empty()
            );
        } else if (
            scenario == ConditioningScenario::failed_second_frequency
            || scenario == ConditioningScenario::failed_second_linear1
        ) {
            check(first.calls == std::vector<char>{'f', 'l', 'l', 't', 't', 'm', 'm'});
            check(
                second.calls
                == (scenario == ConditioningScenario::failed_second_frequency
                        ? std::vector<char>{'f'}
                        : std::vector<char>{'f', 'l'})
            );
        } else {
            check(first.calls.empty() && second.calls.empty());
        }
        check(
            metadata.allocations.empty() && activity->owners == 0
            && conditioning_resources->snapshot().used[1] == 0
            && conditioning_resources->snapshot().used[2] == 0
        );
    };
    conditioning_case(HiddenBoundary::post_attention, ConditioningScenario::complete);
    for (const auto scenario :
         {ConditioningScenario::failed_frequency,
          ConditioningScenario::failed_second_frequency,
          ConditioningScenario::cancelled_frequency,
          ConditioningScenario::failed_linear1,
          ConditioningScenario::failed_second_linear1,
          ConditioningScenario::failed_linear1_row1,
          ConditioningScenario::cancelled_linear1,
          ConditioningScenario::invalid_negative,
          ConditioningScenario::invalid_overshoot,
          ConditioningScenario::invalid_nan,
          ConditioningScenario::invalid_infinity,
          ConditioningScenario::missing_hidden,
          ConditioningScenario::failed_modulation,
          ConditioningScenario::failed_timestep,
          ConditioningScenario::failed_second_timestep,
          ConditioningScenario::failed_timestep_row1,
          ConditioningScenario::supplied_output,
          ConditioningScenario::cancelled_timestep,
          ConditioningScenario::missing_launcher})
        conditioning_case(HiddenBoundary::post_attention, scenario);
    conditioning_case(HiddenBoundary::pre_attention, ConditioningScenario::complete);
    for (const auto scenario :
         {ConditioningScenario::failed_frequency,
          ConditioningScenario::failed_second_frequency,
          ConditioningScenario::cancelled_frequency,
          ConditioningScenario::failed_linear1,
          ConditioningScenario::failed_second_linear1,
          ConditioningScenario::failed_linear1_row1,
          ConditioningScenario::cancelled_linear1,
          ConditioningScenario::invalid_negative,
          ConditioningScenario::invalid_overshoot,
          ConditioningScenario::invalid_nan,
          ConditioningScenario::invalid_infinity,
          ConditioningScenario::missing_hidden,
          ConditioningScenario::missing_attended,
          ConditioningScenario::failed_modulation,
          ConditioningScenario::failed_timestep,
          ConditioningScenario::failed_second_timestep,
          ConditioningScenario::failed_timestep_row1,
          ConditioningScenario::supplied_output,
          ConditioningScenario::cancelled_timestep,
          ConditioningScenario::missing_launcher})
        conditioning_case(HiddenBoundary::pre_attention, scenario);
    conditioning_case(
        HiddenBoundary::post_attention, ConditioningScenario::wrong_boundary
    );
    conditioning_case(
        HiddenBoundary::post_attention, ConditioningScenario::missing_conditioning
    );
    denoiser.unload();

    // Two actual block manifests reuse only session-owned certified conditioning.
    // This fake checks lifetime, identities and submissions, never activation data.
    write_weights(directory, false, 64, 128, {.denoiser = true, .layers = 2});
    auto shared_config = config;
    shared_config.layers = 2;
    denoiser.load(directory, files, shared_config, cancel);
    auto shared_conditioning_reuse = [&] {
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        metadata.metadata_execution = true;
        metadata.available = budget + 78592;
        kadan::MediaSession resident(
            conditioning_resources,
            activity,
            std::move(fake),
            kadan::Workload::image,
            {{0}, {1}},
            cancel
        );
        std::array<void*, 2> time{}, modulation{};
        std::array<std::uint64_t, 2> time_generation{}, modulation_generation{};
        kadan::Bytes shared_bytes = 0;
        for (std::size_t step = 0; step < 2; ++step) {
            for (std::size_t block_index = 0; block_index < 2; ++block_index) {
                check(!denoiser.prepare_feed_forward(
                    resident, block_index, 17, HiddenBoundary::post_attention, cancel
                ));
                check(resident.plan().devices.size() == 2);
                if (!step && !block_index) {
                    for (const auto& w : resident.plan().weights)
                        if (!w.name.starts_with("transformer_blocks."))
                            shared_bytes += w.stored_bytes;
                    for (const auto& d : resident.plan().devices) {
                        auto hidden = resident.tensor(
                            d.device,
                            "resident/hidden",
                            kadan::DeviceDType::f32,
                            {17, 64},
                            {64, 1}
                        );
                        std::vector<std::byte> ingress(hidden.layout().bytes);
                        resident.upload_input(hidden, ingress);
                    }
                }
                const auto uploads = resident.transfers().weight_upload;
                Launcher first(metadata, 0), second(metadata, 1);
                const std::array<kadan::video::ImageFeedForwardLauncher*, 2> launchers{
                    &first, &second
                };
                if (block_index) {
                    first.modulation = modulation[0];
                    first.modulation_generation = modulation_generation[0];
                    second.modulation = modulation[1];
                    second.modulation_generation = modulation_generation[1];
                }

                const float timestep = step ? .25f : .625f;
                first.expected_timestep = second.expected_timestep = timestep;
                denoiser.execute_feed_forward(
                    resident,
                    block_index,
                    17,
                    3,
                    block_index ? std::optional<float>{}
                                : std::optional<float>{timestep},
                    HiddenBoundary::post_attention,
                    launchers,
                    cancel
                );
                check(
                    first.supplied_timestep
                    == (block_index ? std::optional<float>{}
                                    : std::optional<float>{timestep})
                );
                check(first.supplied_timestep == second.supplied_timestep);
                check(resident.transfers().weight_upload == uploads);
                check(first.calls == second.calls);
                check(first.calls.front() == (block_index ? 'n' : 'f'));
                if (block_index)
                    check(std::count(first.calls.begin(), first.calls.end(), 't') == 0);
                for (const auto& d : resident.plan().devices) {
                    auto retained = resident.tensor(
                        d.device,
                        "conditioning/time-activated",
                        kadan::DeviceDType::f32,
                        {2, 64},
                        {64, 1}
                    );
                    check(resident.ready(retained));
                    auto target = resident.rows(retained, 0, 1);
                    const std::array ready{&target};
                    resident.require_ready(ready);
                    if (step || block_index)
                        check(
                            retained.layout().data == time[d.device]
                            && retained.layout().generation == time_generation[d.device]
                        );
                    time[d.device] = retained.layout().data;
                    time_generation[d.device] = retained.layout().generation;
                    retained = {};
                    target = {};
                    auto mod = resident.tensor(
                        d.device,
                        "ff/modulation",
                        kadan::DeviceDType::f32,
                        {2, 256},
                        {256, 1}
                    );
                    check(resident.ready(mod));
                    if (step || block_index)
                        check(
                            mod.layout().data == modulation[d.device]
                            && mod.layout().generation
                                   == modulation_generation[d.device]
                        );
                    modulation[d.device] = mod.layout().data;
                    modulation_generation[d.device] = mod.layout().generation;
                }
            }
        }

        // Only the first admission uploads the three replicated shared weights.
        const auto block_bytes = 2 * (64 * 128 + 128 * 64 + 64 * 128);
        check(resident.transfers().weight_upload == shared_bytes + 4 * block_bytes);
        check(
            !resident.transfers().activation_upload
            && !resident.transfers().activation_download
            && !resident.transfers().output_download
        );
        resident.finish();
        check(metadata.allocations.empty() && activity->owners == 0);
    };
    shared_conditioning_reuse();
    auto missing_retained_conditioning = [&](const char* name, std::size_t columns) {
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        metadata.metadata_execution = true;
        metadata.available = budget + 78592;
        kadan::MediaSession resident(
            conditioning_resources,
            activity,
            std::move(fake),
            kadan::Workload::image,
            {{0}, {1}},
            cancel
        );
        check(!denoiser.prepare_feed_forward(
            resident, 0, 17, HiddenBoundary::post_attention, cancel
        ));
        for (const auto& d : resident.plan().devices) {
            auto hidden = resident.tensor(
                d.device, "resident/hidden", kadan::DeviceDType::f32, {17, 64}, {64, 1}
            );
            std::vector<std::byte> ingress(hidden.layout().bytes);
            resident.upload_input(hidden, ingress);
        }
        Launcher producer0(metadata, 0), producer1(metadata, 1);
        const std::array<kadan::video::ImageFeedForwardLauncher*, 2> producers{
            &producer0, &producer1
        };
        denoiser.execute_feed_forward(
            resident, 0, 17, 3, .625f, HiddenBoundary::post_attention, producers, cancel
        );
        check(!denoiser.prepare_feed_forward(
            resident, 1, 17, HiddenBoundary::post_attention, cancel
        ));
        {
            // Only device 1 loses certification, independently for time2/modulation.
            auto retained = resident.tensor(
                1, name, kadan::DeviceDType::f32, {2, columns}, {columns, 1}
            );
            check(resident.ready(retained));
            retained = {};
            auto changed =
                resident.tensor(1, name, kadan::DeviceDType::f32, {2 * columns}, {1});
        }
        Launcher consumer0(metadata, 0), consumer1(metadata, 1);
        const std::array<kadan::video::ImageFeedForwardLauncher*, 2> consumers{
            &consumer0, &consumer1
        };
        bool rejected = false;
        try {
            denoiser.execute_feed_forward(
                resident,
                1,
                17,
                3,
                std::nullopt,
                HiddenBoundary::post_attention,
                consumers,
                cancel
            );
        } catch (const std::invalid_argument& e) {
            check(std::string_view(e.what()) == "media_session_unready_input");
            rejected = true;
        }
        check(rejected && consumer0.calls.empty() && consumer1.calls.empty());
        check(metadata.allocations.empty() && activity->owners == 0);
    };
    missing_retained_conditioning("conditioning/time-activated", 64);
    missing_retained_conditioning("ff/modulation", 256);
    denoiser.unload();
    for (const int unsupported : {0, 1, 2, 3}) {
        const bool native = unsupported != 0;
        write_weights(
            directory,
            false,
            64,
            128,
            {.denoiser = true,
             .modulation_f32 = unsupported == 1,
             .timestep_f32 = unsupported == 2,
             .linear1_f32 = unsupported == 3}
        );
        kadan::image::Denoiser unsupported_denoiser(
            conditioning_resources, native ? std::make_shared<Compute>() : nullptr
        );
        unsupported_denoiser.load(directory, files, config, cancel);
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        kadan::MediaSession resident(
            conditioning_resources,
            activity,
            std::move(fake),
            kadan::Workload::image,
            {{0}, {1}},
            cancel
        );
        const auto error = unsupported_denoiser.prepare_feed_forward(
            resident, 0, 17, HiddenBoundary::post_attention, cancel
        );
        check(
            error
            && error->detail
                   == (unsupported == 1 ? "image_modulation_standard_precision"
                                        : "image_timestep_standard_precision")
            && !metadata.probes && !metadata.uploaded && metadata.allocations.empty()
        );
        resident.finish();
    }
    for (const int timestep : {0, 1, 2}) {
        write_weights(directory, false, 64, 128, {.denoiser = true});
        {
            // Corrupt an actual shared raw weight before loading; no arithmetic runs.
            std::fstream file(
                std::string(directory) + "/weights.safetensors",
                std::ios::in | std::ios::out | std::ios::binary
            );
            file.seekp(
                -std::streamoff(
                    32768 + 2 * 8192
                    + (timestep == 2   ? 8192 + 32768
                       : timestep == 1 ? 8192
                                       : 0)
                ),
                std::ios::end
            );
            file.put(char(0x80));
            file.put(char(0x7f));
            check(bool(file));
        }
        denoiser.load(directory, files, config, cancel);
        {
            auto fake = std::make_unique<Device>();
            auto& metadata = *fake;
            metadata.metadata_execution = true;
            kadan::MediaSession resident(
                conditioning_resources,
                activity,
                std::move(fake),
                kadan::Workload::image,
                {{0}, {1}},
                cancel
            );
            bool rejected = false;
            try {
                denoiser.prepare_feed_forward(
                    resident, 0, 17, HiddenBoundary::post_attention, cancel
                );
            } catch (const std::runtime_error& e) {
                rejected = true;
                check(std::string_view(e.what()) == "h3_cuda_nonfinite_weight");
            }
            check(rejected);
            check(metadata.allocations.empty());
            check(activity->owners == 0);
        }
        denoiser.unload();
    }
    check(conditioning_resources->snapshot().used == kadan::Footprint({0, 0, 0}));
    write_weights(directory, true);
    block.load(directory, "weights.safetensors", 7, {2, 1, 2, 2, {0, 0, 2}}, cancel);
    auto unsupported_backend = std::make_unique<Device>();
    auto& unsupported_device = *unsupported_backend;
    kadan::MediaSession unsupported(
        resources,
        activity,
        std::move(unsupported_backend),
        kadan::Workload::image,
        {{0}},
        cancel
    );
    const auto error = block.prepare_feed_forward(
        unsupported, 5, HiddenBoundary::post_attention, cancel
    );
    check(error && error->reason == kadan::PlanFailure::unsupported_layout);
    check(!unsupported_device.probes && !unsupported_device.uploaded);
    check(unsupported_device.allocations.empty());
    unsupported.finish();
    block.unload();
    check(resources->snapshot().used == kadan::Footprint({0, 0}));
    auto unsupported_attention = [&](bool native, bool attention_f32) {
        write_weights(directory, false, 2, 2, {.attention_f32 = attention_f32});
        kadan::image::TransformerBlock standard_block(
            resources, native ? std::make_shared<Compute>() : nullptr
        );
        standard_block.load(
            directory, "weights.safetensors", 7, {2, 1, 2, 2, {0, 0, 2}}, cancel
        );
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        kadan::MediaSession resident(
            resources, activity, std::move(fake), kadan::Workload::image, {{0}}, cancel
        );
        const auto rejected = standard_block.prepare_feed_forward(
            resident, 5, HiddenBoundary::pre_attention, cancel
        );
        check(
            rejected && rejected->reason == kadan::PlanFailure::unsupported_layout
            && rejected->detail == "image_attention_standard_precision"
        );
        check(!metadata.probes && !metadata.uploaded && metadata.allocations.empty());
        resident.finish();
    };
    unsupported_attention(false, false);
    unsupported_attention(true, true);
    check(resources->snapshot().used == kadan::Footprint({0, 0}));
    std::cout << "qualified block checkpoint names and release passed (no execution)\n";
}
