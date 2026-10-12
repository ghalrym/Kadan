// Checkpoint ingress/ownership with a fake device; no CUDA or projection execution.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unistd.h>

#include "kadan/bf16_projection.hpp"
#include "kadan/image.hpp"

void check(bool b) {
    if (!b)
        throw std::runtime_error("test_failed");
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

void write_weights(
    const char* directory,
    bool f32,
    std::size_t state = 2,
    std::size_t ff = 2,
    bool attention_f32 = false
) {
    std::ostringstream header;
    header << '{';
    std::size_t bytes = 0, count = 0;
    auto add = [&](std::string name, bool vector) {
        if (count++)
            header << ',';
        const auto rows =
            name == "img_mlp.proj" || name == "img_mlp.gate_layer" ? ff : state;
        const auto columns = name == "img_mlp.out" ? ff : state;
        const bool tensor_f32 = f32 || (attention_f32 && name == "attn.to_out.0");
        const auto size = (vector ? state : rows * columns) * (tensor_f32 ? 4 : 2);
        header << "\"transformer_blocks.7." << name << ".weight\":{\"dtype\":\""
               << (tensor_f32 ? "F32" : "BF16") << "\",\"shape\":["
               << (vector ? state : rows);
        if (!vector)
            header << ',' << columns;
        header << "],\"data_offsets\":[" << bytes << ',' << bytes + size << "]}";
        bytes += size;
    };
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
         fail_projection = false;
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

    void attention_output(
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
        check(std::all_of(calls.begin(), calls.end(), [](char c) { return c == 'o'; }));
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
        check(calls.empty() || calls.back() == 'a');
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
        write_weights(directory, false, 2, 2, attention_f32);
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
