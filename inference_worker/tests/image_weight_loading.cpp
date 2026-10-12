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
#include "kadan/checkpoint.hpp"
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
         timestep_f32 = false, linear1_f32 = false, final_f32 = false,
         output_f32 = false, norm_f32 = false;
    const char* ingress_f32 = "";
    const char* qkv_f32 = "";
    std::size_t context = 0;
    std::size_t layers = 1, channels = 64;
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
        const bool tensor_f32 = f32
                                || (options.attention_f32 && name == "attn.to_out.0")
                                || name == options.qkv_f32;
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
                           == "time_text_embed.timestep_embedder.linear_1")
                || (options.final_f32 && std::string_view(name) == "norm_out.linear")
                || (options.output_f32 && std::string_view(name) == "proj_out")
                || (std::string_view(name) == options.ingress_f32)
                || (options.norm_f32 && std::string_view(name) == "txt_in.text_norm");
            const auto size = rows * (columns ? columns : 1) * (fp32 ? 4 : 2);
            header << ",\"" << name << ".weight\":{\"dtype\":\""
                   << (fp32 ? "F32" : "BF16") << "\",\"shape\":[" << rows;
            if (columns)
                header << ',' << columns;
            header << "],\"data_offsets\":[" << bytes << ',' << bytes + size << "]}";
            bytes += size;
        };
        shared("img_in", state, options.channels);
        const auto context = options.context ? options.context : state;
        shared("txt_in.text_norm", context, 0);
        shared("txt_in.in_layer", state, context);
        shared("txt_in.out_layer", state, state);
        shared("time_text_embed.timestep_embedder.linear_1", state, 256);
        shared("time_text_embed.timestep_embedder.linear_2", state, state);
        shared("modulation.1", 4 * state, state);
        shared("norm_out.linear", state, state);
        shared("proj_out", options.channels, state);
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
    bool ingress = false;
    std::size_t ingress_text = 3, ingress_context = 64;
    char fail_ingress = 0, cancel_ingress_stage = 0;
    std::atomic_bool* cancel_ingress = nullptr;
    void* ingress_normalized = nullptr;
    std::uint64_t ingress_normalized_generation = 0;
    void* ingress_projected = nullptr;
    std::uint64_t ingress_projected_generation = 0;
    void* ingress_hidden = nullptr;
    std::uint64_t ingress_hidden_generation = 0;
    std::map<std::string, kadan::MediaDeviceSpan> ingress_weights;
    std::vector<std::size_t> text_rows, image_rows;
    bool qkv = false;
    std::size_t qkv_projection = 0;
    char fail_qkv = 0, cancel_qkv_stage = 0;
    std::atomic_bool* cancel_qkv = nullptr;
    void* qkv_normalized = nullptr;
    std::uint64_t qkv_normalized_generation = 0;
    std::map<char, kadan::MediaDeviceSpan> qkv_weights, qkv_outputs;
    std::map<char, std::vector<std::size_t>> qkv_rows;
    bool fail_pair = false, fail_preparation = false, fail_attention = false,
         fail_projection = false, fail_modulation = false;
    bool fail_timestep = false, fail_timestep_row1 = false;
    bool fail_frequency = false, fail_linear1 = false, fail_linear1_row1 = false;
    bool fail_final_projection = false, fail_final_normalization = false,
         fail_velocity = false;
    std::size_t channels = 64;
    std::atomic_bool *cancel_final_normalization = nullptr, *cancel_velocity = nullptr;
    void* output_weight = nullptr;
    std::uint64_t output_weight_generation = 0;
    void* final_normalized = nullptr;
    std::uint64_t final_normalized_generation = 0;
    std::atomic_bool* cancel_final_projection = nullptr;
    void* final_hidden = nullptr;
    std::uint64_t final_hidden_generation = 0;
    void* final_scale = nullptr;
    void* final_weight = nullptr;
    std::uint64_t final_weight_generation = 0;
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

    void expect_ingress_weights(kadan::MediaSession& session) {
        for (const auto& w : session.plan().weights) {
            if (w.device != selected
                || !(w.name.starts_with("txt_in.") || w.name == "img_in.weight"))
                continue;
            check(w.first_row == 0 && w.identity.inode != 0);
            auto view = session.weight(w);
            ingress_weights.emplace(w.name, view.layout());
        }
        check(ingress_weights.size() == 4);
    }

    void ingress_status(char stage) {
        calls.push_back(stage);
        backend.status = fail_ingress == stage ? 1 : 0;
        if (cancel_ingress && cancel_ingress_stage == stage)
            cancel_ingress->store(true);
    }

    void check_ingress_weight(const kadan::MediaDeviceSpan& span, const char* name) {
        const auto& expected = ingress_weights.at(name);
        check(
            span.data == expected.data && span.generation == expected.generation
            && span.dimensions == expected.dimensions && span.dtype == expected.dtype
        );
    }

    void text_normalization(
        void*,
        std::span<const kadan::MediaDeviceSpan> read,
        std::span<const kadan::MediaDeviceSpan> write
    ) override {
        check(ingress && read.size() == 2 && write.size() == 2 && calls.empty());
        kadan::video::validate_image_text_normalization(
            selected, read[0], read[1], write[0], write[1]
        );
        check(
            read[0].dimensions
            == std::vector<std::size_t>{ingress_text, ingress_context}
        );
        check_ingress_weight(read[1], "txt_in.text_norm.weight");
        ingress_normalized = write[0].data;
        ingress_normalized_generation = write[0].generation;
        ingress_status('N');
    }

    void text_projection(
        void*,
        std::span<const kadan::MediaDeviceSpan> read,
        std::span<const kadan::MediaDeviceSpan> write
    ) override {
        check(ingress && read.size() == 2 && write.size() == 4);
        const kadan::MediaDeviceSpan bias{
            nullptr, selected, kadan::DeviceDType::f32, {}, {}, 0, 0, 0
        };
        kadan::video::validate_bf16_projection(
            selected, read[0], read[1], bias, write[1], write[0], write[2], write[3]
        );
        check_ingress_weight(read[1], "txt_in.in_layer.weight");
        check(
            static_cast<std::byte*>(read[0].data) - read[0].offset == ingress_normalized
            && read[0].generation == ingress_normalized_generation
        );
        check(read[0].offset / (ingress_context * 4) == text_rows.size());
        text_rows.push_back(read[0].offset / (ingress_context * 4));
        ingress_projected = static_cast<std::byte*>(write[0].data) - write[0].offset;
        ingress_projected_generation = write[0].generation;
        ingress_status('G');
    }

    void expect_final_weight(kadan::MediaSession& session) {
        const auto& weights = session.plan().weights;
        const auto found =
            std::find_if(weights.begin(), weights.end(), [&](const auto& w) {
                return w.device == selected && w.name == "norm_out.linear.weight";
            });
        check(found != weights.end() && found->identity.inode != 0);
        auto view = session.weight(*found);
        final_weight = view.layout().data;
        final_weight_generation = view.layout().generation;
        const auto output =
            std::find_if(weights.begin(), weights.end(), [&](const auto& w) {
                return w.device == selected && w.name == "proj_out.weight";
            });
        check(
            output != weights.end() && output->identity.inode != 0
            && output->rows == channels && output->first_row == 0
            && output->stored_bytes == channels * 64 * 2
        );
        auto output_view = session.weight(*output);
        output_weight = output_view.layout().data;
        output_weight_generation = output_view.layout().generation;
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
        if (ingress) {
            const bool image = read[1].data == ingress_weights.at("img_in.weight").data;
            check_ingress_weight(
                read[1], image ? "img_in.weight" : "txt_in.out_layer.weight"
            );
            check(write[0].dimensions == std::vector<std::size_t>{1, 64});
            const auto row = write[0].offset / 256;
            if (image) {
                check(row == ingress_text + image_rows.size());
                check(read[0].dimensions == std::vector<std::size_t>{1, channels});
                image_rows.push_back(row);
            } else {
                check(text_rows.size() == ingress_text && row < ingress_text);
                check(
                    static_cast<std::byte*>(read[0].data) - read[0].offset
                        == ingress_projected
                    && read[0].generation == ingress_projected_generation
                );
            }
            const auto hidden =
                static_cast<std::byte*>(write[0].data) - write[0].offset;
            if (ingress_hidden)
                check(
                    hidden == ingress_hidden
                    && write[0].generation == ingress_hidden_generation
                );
            ingress_hidden = hidden;
            ingress_hidden_generation = write[0].generation;
            ingress_status(image ? 'I' : 'T');
            return;
        }
        if (qkv && !calls.empty()
            && (calls.back() == '0' || calls.back() == 'q' || calls.back() == 'k'
                || calls.back() == 'v')) {
            const std::array stages{'q', 'k', 'v'};
            check(qkv_projection < stages.size());
            const auto stage = stages[qkv_projection];
            const auto& weight = qkv_weights.at(stage);
            check(
                read[1].data == weight.data && read[1].generation == weight.generation
            );
            check(
                static_cast<std::byte*>(read[0].data) - read[0].offset == qkv_normalized
                && read[0].generation == qkv_normalized_generation
            );
            const auto row = read[0].offset / 256;
            check(row == qkv_rows[stage].size() && write[0].offset == read[0].offset);
            qkv_rows[stage].push_back(row);
            if (row == 0)
                qkv_outputs[stage] = write[0];
            else
                check(
                    static_cast<std::byte*>(write[0].data) - write[0].offset
                        == qkv_outputs.at(stage).data
                    && write[0].generation == qkv_outputs.at(stage).generation
                );
            if (row == 16)
                ++qkv_projection;
            calls.push_back(stage);
            backend.status = fail_qkv == stage ? 1 : 0;
            if (cancel_qkv && cancel_qkv_stage == stage)
                cancel_qkv->store(true);
            return;
        }
        if (!calls.empty() && calls.back() == 'z') {
            check(
                read[0].dimensions == std::vector<std::size_t>{14, 64}
                && read[0].offset == 0 && read[0].data == final_normalized
                && read[0].generation == final_normalized_generation
            );
            check(
                read[1].dimensions == std::vector<std::size_t>{channels, 64}
                && read[1].data == output_weight
                && read[1].generation == output_weight_generation
            );
            check(
                write[0].dimensions == std::vector<std::size_t>{14, channels}
                && write[1].dimensions == std::vector<std::size_t>{14, 64}
            );
            calls.push_back('v');
            backend.status = fail_velocity ? 1 : 0;
            if (cancel_velocity)
                cancel_velocity->store(true);
            return;
        }
        if (!calls.empty() && calls.back() == 'r') {
            check(read[0].dimensions == std::vector<std::size_t>{1, 64});
            check(
                read[0].data == activated && read[0].offset == 0
                && read[0].generation == activated_generation
            );
            check(read[1].dimensions == std::vector<std::size_t>{64, 64});
            check(write[0].dimensions == std::vector<std::size_t>{1, 64});
            check(
                read[1].data == final_weight
                && read[1].generation == final_weight_generation
            );
            final_scale = write[0].data;
            calls.push_back('s');
            backend.status = fail_final_projection ? 1 : 0;
            if (cancel_final_projection)
                cancel_final_projection->store(true);
            return;
        }
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
        if (ingress_hidden)
            check(
                read[0].data == ingress_hidden
                && read[0].generation == ingress_hidden_generation
            );
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
        std::size_t text,
        kadan::video::ImageModulationStage stage
    ) override {
        check(
            stage
            == (qkv ? kadan::video::ImageModulationStage::qkv
                    : kadan::video::ImageModulationStage::feed_forward)
        );
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
        if (qkv) {
            check(
                read[0].data == ingress_hidden
                && read[0].generation == ingress_hidden_generation
            );
            qkv_normalized = write[0].data;
            qkv_normalized_generation = write[0].generation;
            calls.push_back('0');
            backend.status = fail_qkv == '0' ? 1 : 0;
            if (cancel_qkv && cancel_qkv_stage == '0')
                cancel_qkv->store(true);
        } else {
            calls.push_back('n');
            backend.status = fail_preparation ? 1 : 0;
        }
    }

    void final_normalization(
        void*,
        std::span<const kadan::MediaDeviceSpan> read,
        std::span<const kadan::MediaDeviceSpan> write
    ) override {
        check(read.size() == 2 && write.size() == 2 && calls.back() == 's');
        kadan::video::validate_image_final_normalization(
            selected, read[0], read[1], write[0], write[1]
        );
        check(
            read[0].dimensions == std::vector<std::size_t>{14, 64}
            && read[0].offset == 3 * 64 * 4
        );
        check(
            static_cast<std::byte*>(read[0].data) - read[0].offset == final_hidden
            && read[0].generation == final_hidden_generation
        );
        check(
            read[1].data == final_scale
            && read[1].dimensions == std::vector<std::size_t>{1, 64}
        );
        check(write[0].dimensions == std::vector<std::size_t>{14, 64});
        final_normalized = write[0].data;
        final_normalized_generation = write[0].generation;
        calls.push_back('z');
        backend.status = fail_final_normalization ? 1 : 0;
        if (cancel_final_normalization)
            cancel_final_normalization->store(true);
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
        final_hidden = static_cast<std::byte*>(hidden.data) - hidden.offset;
        final_hidden_generation = hidden.generation;
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
    write_weights(directory, false, 64, 512, {.denoiser = true});
    // The wider fake FF keeps both early and final manifests on two devices,
    // with room for retained owners and complete-phase output allocations.
    constexpr kadan::Bytes conditioning_budget = budget + 230000;
    auto conditioning_resources = std::make_shared<kadan::Resources>(
        kadan::Footprint{128 * 1024 * 1024, conditioning_budget, conditioning_budget}
    );
    kadan::image::Denoiser denoiser(
        conditioning_resources, std::make_shared<Compute>()
    );
    const std::array<std::string, 1> files{"weights.safetensors"};
    const kadan::image::DenoiserConfig config{{64, 1, 64, 512, {0, 0, 64}}, 1, 64, 64};
    denoiser.load(directory, files, config, cancel);
    // Actual ingress producers are metadata fakes: no activation reads/arithmetic.
    enum class IngressScenario {
        complete,
        fail_norm,
        fail_gelu,
        fail_text,
        fail_image,
        cancel_norm,
        cancel_gelu,
        cancel_text,
        cancel_image,
        missing_launcher,
        invalid_condition,
        invalid_latent,
        nonfinite_condition,
        nonfinite_latent
    };
    // Each fake validates metadata/certificates; none reads an activation.
    enum class QkvScenario {
        complete,
        fail_norm,
        fail_q,
        fail_k,
        fail_v,
        cancel_norm,
        cancel_q,
        cancel_k,
        cancel_v,
        missing_hidden,
        missing_launcher,
        invalid_text,
        invalid_tokens,
        wrong_boundary
    };
    auto ingress_case = [&](IngressScenario scenario,
                            bool two_devices,
                            std::optional<QkvScenario> qkv_scenario = {}) {
        const auto ingress_budget = conditioning_budget + (qkv_scenario ? 40000 : 0);
        auto ingress_resources = std::make_shared<kadan::Resources>(
            two_devices
                ? kadan::Footprint{128 * 1024 * 1024, ingress_budget, ingress_budget}
                : kadan::Footprint{128 * 1024 * 1024, 32 * 1024 * 1024}
        );
        kadan::image::Denoiser ingress_denoiser(
            ingress_resources, std::make_shared<Compute>()
        );
        auto ingress_config = config;
        ingress_config.context = 96;
        ingress_config.channels = 32;
        write_weights(
            directory,
            false,
            64,
            512,
            {.denoiser = true, .norm_f32 = !two_devices, .context = 96, .channels = 32}
        );
        ingress_denoiser.load(directory, files, ingress_config, cancel);
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        metadata.available = two_devices ? ingress_budget : 32 * 1024 * 1024;
        metadata.metadata_execution = true;
        kadan::MediaSession resident(
            ingress_resources,
            activity,
            std::move(fake),
            kadan::Workload::image,
            two_devices ? std::vector<kadan::MediaDeviceLimit>{{0}, {1}}
                        : std::vector<kadan::MediaDeviceLimit>{{0}},
            cancel
        );
        if (const auto error = ingress_denoiser.prepare_feed_forward(
                resident,
                0,
                17,
                qkv_scenario && *qkv_scenario != QkvScenario::wrong_boundary
                    ? HiddenBoundary::pre_qkv
                    : HiddenBoundary::pre_attention,
                cancel
            ))
            throw std::runtime_error("ingress_admission:" + error->detail);
        check(resident.plan().devices.size() == (two_devices ? 2 : 1));
        Launcher first(metadata, 0), second(metadata, 1);
        for (auto* launcher : {&first, &second}) {
            launcher->ingress = true;
            launcher->ingress_context = 96;
            launcher->channels = 32;
            launcher->cancel_ingress = &cancel;
        }
        for (const auto& d : resident.plan().devices)
            (d.device == 0 ? first : second).expect_ingress_weights(resident);
        auto& target = two_devices ? second : first;
        if (scenario == IngressScenario::fail_norm)
            target.fail_ingress = 'N';
        if (scenario == IngressScenario::fail_gelu)
            target.fail_ingress = 'G';
        if (scenario == IngressScenario::fail_text)
            target.fail_ingress = 'T';
        if (scenario == IngressScenario::fail_image)
            target.fail_ingress = 'I';
        if (scenario == IngressScenario::cancel_norm)
            target.cancel_ingress_stage = 'N';
        if (scenario == IngressScenario::cancel_gelu)
            target.cancel_ingress_stage = 'G';
        if (scenario == IngressScenario::cancel_text)
            target.cancel_ingress_stage = 'T';
        if (scenario == IngressScenario::cancel_image)
            target.cancel_ingress_stage = 'I';
        std::vector<kadan::video::ImageFeedForwardLauncher*> launchers{&first};
        if (two_devices)
            launchers.push_back(&second);
        if (scenario == IngressScenario::missing_launcher)
            launchers.back() = nullptr;
        std::vector<float> latent(14 * 32), condition(3 * 96);
        if (scenario == IngressScenario::invalid_condition)
            condition.pop_back();
        if (scenario == IngressScenario::invalid_latent)
            latent.pop_back();
        if (scenario == IngressScenario::nonfinite_condition)
            condition[0] = std::numeric_limits<float>::infinity();
        if (scenario == IngressScenario::nonfinite_latent)
            latent[0] = std::numeric_limits<float>::quiet_NaN();
        bool rejected = false;
        try {
            ingress_denoiser.execute_ingress(
                resident, latent, condition, 3, launchers, cancel
            );
        } catch (const std::exception& e) {
            rejected = true;
            const auto error = std::string_view(e.what());
            check(
                error
                == (target.fail_ingress           ? "media_session_nonfinite_output"
                    : target.cancel_ingress_stage ? "media_session_cancelled"
                    : scenario == IngressScenario::missing_launcher
                        ? "image_ingress_launcher"
                    : (scenario == IngressScenario::nonfinite_condition
                       || scenario == IngressScenario::nonfinite_latent)
                        ? "image_nonfinite"
                        : "image_ingress_shape")
            );
        }
        check(rejected == (scenario != IngressScenario::complete));
        cancel.store(false);
        if (!rejected) {
            check(
                resident.transfers().input_upload
                == (latent.size() + condition.size()) * 4 * launchers.size()
            );
            for (const auto& d : resident.plan().devices) {
                auto& launcher = d.device == 0 ? first : second;
                auto hidden = resident.tensor(
                    d.device,
                    "resident/hidden",
                    kadan::DeviceDType::f32,
                    {17, 64},
                    {64, 1}
                );
                check(
                    resident.ready(hidden)
                    && hidden.layout().data == launcher.ingress_hidden
                    && hidden.layout().generation == launcher.ingress_hidden_generation
                );
                check(launcher.calls.size() == 21 && launcher.calls.front() == 'N');
                check(launcher.text_rows == std::vector<std::size_t>{0, 1, 2});
                check(
                    launcher.image_rows.front() == 3 && launcher.image_rows.back() == 16
                );
                if (qkv_scenario)
                    continue;
                // The existing block consumer binds this exact certified owner.
                std::vector<std::byte> attended(17 * 64 * 4);
                auto normalized = resident.tensor(
                    d.device,
                    "resident/normalized",
                    kadan::DeviceDType::f32,
                    {17, 64},
                    {64, 1}
                );
                resident.upload_input(normalized, attended);
                launcher.ingress = false;
                launcher.calls.clear();
                launcher.expect_final_weight(resident);
            }
            if (qkv_scenario) {
                std::vector<Launcher*> qkv_launchers{&first};
                if (two_devices)
                    qkv_launchers.push_back(&second);
                for (auto* launcher : qkv_launchers) {
                    launcher->ingress = false;
                    launcher->qkv = true;
                    launcher->calls.clear();
                    launcher->cancel_qkv = &cancel;
                    for (const auto& w : resident.plan().weights) {
                        if (w.device != launcher->device())
                            continue;
                        for (const auto stage : {'q', 'k', 'v'})
                            if (w.name
                                == std::string("transformer_blocks.0.attn.to_") + stage
                                       + ".weight") {
                                check(
                                    w.first_row == 0 && w.rows == 64 && w.identity.inode
                                );
                                auto view = resident.weight(w);
                                launcher->qkv_weights[stage] = view.layout();
                            }
                    }
                }
                const auto mode = *qkv_scenario;
                const std::array stages{'0', 'q', 'k', 'v'};
                if (mode >= QkvScenario::fail_norm && mode <= QkvScenario::fail_v)
                    target.fail_qkv =
                        stages[std::size_t(mode) - std::size_t(QkvScenario::fail_norm)];
                if (mode >= QkvScenario::cancel_norm && mode <= QkvScenario::cancel_v)
                    target.cancel_qkv_stage = stages
                        [std::size_t(mode) - std::size_t(QkvScenario::cancel_norm)];
                if (mode == QkvScenario::missing_hidden) {
                    auto hidden = resident.tensor(
                        target.device(),
                        "resident/hidden",
                        kadan::DeviceDType::f32,
                        {17, 64},
                        {64, 1}
                    );
                    const std::array<kadan::DeviceTensorView*, 0> read{};
                    const std::array write{&hidden};
                    resident.launch(
                        target.device(), read, write, [](void*, auto, auto) {}
                    );  // Deliberately uncertified producer.
                }
                if (mode == QkvScenario::missing_launcher)
                    launchers.back() = nullptr;
                bool qkv_rejected = false;
                try {
                    ingress_denoiser.execute_qkv(
                        resident,
                        0,
                        mode == QkvScenario::invalid_tokens ? 18 : 17,
                        mode == QkvScenario::invalid_text ? 18 : 3,
                        .625f,
                        launchers,
                        cancel
                    );
                } catch (const std::exception&) {
                    qkv_rejected = true;
                }
                check(qkv_rejected == (mode != QkvScenario::complete));
                cancel.store(false);
                if (mode == QkvScenario::complete) {
                    for (auto* launcher : qkv_launchers)
                        for (const auto stage : {'q', 'k', 'v'}) {
                            const auto name = std::string("attention/") + stage;
                            auto output = resident.tensor(
                                launcher->device(),
                                name,
                                kadan::DeviceDType::f32,
                                {17, 64},
                                {64, 1}
                            );
                            check(resident.ready(output));
                            check(
                                output.layout().data
                                    == launcher->qkv_outputs.at(stage).data
                                && output.layout().generation
                                       == launcher->qkv_outputs.at(stage).generation
                            );
                            check(
                                launcher->qkv_rows.at(stage).size() == 17
                                && launcher->qkv_rows.at(stage).back() == 16
                            );
                        }
                    for (auto* launcher : qkv_launchers) {
                        launcher->calls.clear();
                        launcher->qkv_rows.clear();
                        launcher->qkv_projection = 0;
                    }
                    const auto input_uploads = resident.transfers().input_upload;
                    ingress_denoiser.execute_qkv(
                        resident, 0, 17, 3, std::nullopt, launchers, cancel
                    );
                    check(resident.transfers().input_upload == input_uploads);
                    for (auto* launcher : qkv_launchers)
                        check(
                            launcher->calls.front() == '0'
                            && launcher->qkv_rows.at('v').size() == 17
                        );
                    resident.finish();
                } else if (mode >= QkvScenario::missing_hidden) {
                    check(first.calls.empty() && second.calls.empty());
                } else {
                    check(
                        target.calls.back()
                        == (target.fail_qkv ? target.fail_qkv : target.cancel_qkv_stage)
                    );
                }
                check(
                    !resident.transfers().activation_upload
                    && !resident.transfers().activation_download
                    && !resident.transfers().output_download
                );
            } else {
                ingress_denoiser.execute_feed_forward(
                    resident,
                    0,
                    17,
                    3,
                    .625f,
                    HiddenBoundary::pre_attention,
                    launchers,
                    cancel
                );
            }
        } else if (scenario >= IngressScenario::missing_launcher) {
            check(
                first.calls.empty() && second.calls.empty()
                && !resident.transfers().input_upload
            );
        }
        if (!rejected && !qkv_scenario)
            resident.finish();
        if (target.fail_ingress || target.cancel_ingress_stage)
            check(
                target.calls.back()
                == (target.fail_ingress ? target.fail_ingress
                                        : target.cancel_ingress_stage)
            );
        check(metadata.allocations.empty() && activity->owners == 0);
        check(ingress_resources->snapshot().used[1] == 0);
        if (two_devices)
            check(ingress_resources->snapshot().used[2] == 0);
    };
    for (bool two_devices : {false, true})
        for (auto scenario :
             {IngressScenario::complete,
              IngressScenario::fail_norm,
              IngressScenario::fail_gelu,
              IngressScenario::fail_text,
              IngressScenario::fail_image,
              IngressScenario::cancel_norm,
              IngressScenario::cancel_gelu,
              IngressScenario::cancel_text,
              IngressScenario::cancel_image,
              IngressScenario::missing_launcher,
              IngressScenario::invalid_condition,
              IngressScenario::invalid_latent,
              IngressScenario::nonfinite_condition,
              IngressScenario::nonfinite_latent})
            ingress_case(scenario, two_devices);
    for (bool two_devices : {false, true})
        for (auto scenario :
             {QkvScenario::complete,
              QkvScenario::fail_norm,
              QkvScenario::fail_q,
              QkvScenario::fail_k,
              QkvScenario::fail_v,
              QkvScenario::cancel_norm,
              QkvScenario::cancel_q,
              QkvScenario::cancel_k,
              QkvScenario::cancel_v,
              QkvScenario::missing_hidden,
              QkvScenario::missing_launcher,
              QkvScenario::invalid_text,
              QkvScenario::invalid_tokens,
              QkvScenario::wrong_boundary})
            ingress_case(IngressScenario::complete, two_devices, scenario);
    // Restore the original loaded model's checkpoint before conditioning checks.
    denoiser.unload();
    write_weights(directory, false, 64, 512, {.denoiser = true});
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
        missing_conditioning,
        failed_final_projection,
        failed_final_normalization,
        cancelled_final_projection,
        cancelled_final_normalization,
        failed_velocity,
        failed_second_velocity,
        cancelled_velocity
    };
    auto conditioning_case = [&](HiddenBoundary boundary,
                                 ConditioningScenario scenario) {
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        metadata.available = conditioning_budget;
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
        first.expect_final_weight(resident);
        second.expect_final_weight(resident);
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
        first.fail_final_projection =
            scenario == ConditioningScenario::failed_final_projection;
        first.fail_final_normalization =
            scenario == ConditioningScenario::failed_final_normalization;
        if (scenario == ConditioningScenario::cancelled_final_projection)
            first.cancel_final_projection = &cancel;
        first.cancel_final_normalization =
            scenario == ConditioningScenario::cancelled_final_normalization ? &cancel
                                                                            : nullptr;
        first.fail_velocity = scenario == ConditioningScenario::failed_velocity;
        second.fail_velocity = scenario == ConditioningScenario::failed_second_velocity;
        first.cancel_velocity =
            scenario == ConditioningScenario::cancelled_velocity ? &cancel : nullptr;
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
            auto final = resident.tensor(
                1, "final/normalized", kadan::DeviceDType::f32, {14, 64}, {64, 1}
            );
            auto velocity = resident.tensor(
                1, "final/velocity", kadan::DeviceDType::f32, {14, 64}, {64, 1}
            );
            const std::array ready{&hidden, &final, &velocity};
            resident.require_ready(ready);
            check(first.calls.back() == 'v' && second.calls.back() == 'v');
            for (const auto& live : resident.plan().tensors) {
                if (live.name.starts_with("final/"))
                    check(
                        live.first == kadan::MediaPhase::complete
                        && live.last == kadan::MediaPhase::complete
                    );
            }
            velocity = {};
            final = {};
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
        } else if (
            scenario == ConditioningScenario::failed_final_projection
            || scenario == ConditioningScenario::cancelled_final_projection
            || scenario == ConditioningScenario::failed_final_normalization
            || scenario == ConditioningScenario::cancelled_final_normalization
            || scenario == ConditioningScenario::failed_velocity
            || scenario == ConditioningScenario::cancelled_velocity
        ) {
            char expected_stage = 's';
            if (scenario == ConditioningScenario::failed_final_normalization
                || scenario == ConditioningScenario::cancelled_final_normalization)
                expected_stage = 'z';
            if (scenario == ConditioningScenario::failed_velocity
                || scenario == ConditioningScenario::cancelled_velocity)
                expected_stage = 'v';
            check(first.calls.back() == expected_stage);
            check(second.calls.back() == 'r');
        } else if (scenario == ConditioningScenario::failed_second_velocity) {
            check(first.calls.back() == 'v' && second.calls.back() == 'v');
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
          ConditioningScenario::missing_launcher,
          ConditioningScenario::failed_final_projection,
          ConditioningScenario::failed_final_normalization,
          ConditioningScenario::cancelled_final_projection,
          ConditioningScenario::cancelled_final_normalization,
          ConditioningScenario::failed_velocity,
          ConditioningScenario::failed_second_velocity,
          ConditioningScenario::cancelled_velocity})
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
          ConditioningScenario::missing_launcher,
          ConditioningScenario::failed_final_projection,
          ConditioningScenario::failed_final_normalization,
          ConditioningScenario::cancelled_final_projection,
          ConditioningScenario::cancelled_final_normalization,
          ConditioningScenario::failed_velocity,
          ConditioningScenario::failed_second_velocity,
          ConditioningScenario::cancelled_velocity})
        conditioning_case(HiddenBoundary::pre_attention, scenario);
    conditioning_case(
        HiddenBoundary::post_attention, ConditioningScenario::wrong_boundary
    );
    conditioning_case(
        HiddenBoundary::post_attention, ConditioningScenario::missing_conditioning
    );
    {
        // Automatic placement uses one device when the actual final manifest fits.
        auto single_resources = std::make_shared<kadan::Resources>(
            kadan::Footprint{128 * 1024 * 1024, 32 * 1024 * 1024}
        );
        kadan::image::Denoiser single(single_resources, std::make_shared<Compute>());
        write_weights(directory, false, 64, 512, {.denoiser = true, .channels = 32});
        auto single_config = config;
        single_config.channels = 32;
        single.load(directory, files, single_config, cancel);
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        metadata.metadata_execution = true;
        kadan::MediaSession resident(
            single_resources,
            activity,
            std::move(fake),
            kadan::Workload::image,
            {{0}},
            cancel
        );
        check(!single.prepare_feed_forward(
            resident, 0, 17, HiddenBoundary::post_attention, cancel
        ));
        check(
            resident.plan().devices.size() == 1
            && resident.plan().devices[0].device == 0
        );
        {
            auto hidden = resident.tensor(
                0, "resident/hidden", kadan::DeviceDType::f32, {17, 64}, {64, 1}
            );
            std::vector<std::byte> ingress(hidden.layout().bytes);
            resident.upload_input(hidden, ingress);
        }
        Launcher launcher(metadata, 0);
        launcher.channels = 32;
        launcher.expect_final_weight(resident);
        const std::array<kadan::video::ImageFeedForwardLauncher*, 1> launchers{
            &launcher
        };
        single.execute_feed_forward(
            resident, 0, 17, 3, .625f, HiddenBoundary::post_attention, launchers, cancel
        );
        check(launcher.calls.back() == 'v');
        {
            auto output = resident.tensor(
                0, "final/velocity", kadan::DeviceDType::f32, {14, 32}, {32, 1}
            );
            check(resident.ready(output));
        }
        check(
            !resident.transfers().activation_upload
            && !resident.transfers().activation_download
            && !resident.transfers().peer_activation
            && !resident.transfers().output_download
        );
        resident.finish();
        single.unload();
        check(
            metadata.allocations.empty() && activity->owners == 0
            && single_resources->snapshot().used == kadan::Footprint({0, 0})
        );
    }
    denoiser.unload();

    // Two actual block manifests reuse only session-owned certified conditioning.
    // This fake checks lifetime, identities and submissions, never activation data.
    write_weights(directory, false, 64, 512, {.denoiser = true, .layers = 2});
    auto shared_config = config;
    shared_config.layers = 2;
    denoiser.load(directory, files, shared_config, cancel);
    auto shared_conditioning_reuse = [&] {
        auto fake = std::make_unique<Device>();
        auto& metadata = *fake;
        metadata.metadata_execution = true;
        metadata.available = conditioning_budget;
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
                    first.expect_final_weight(resident);
                    second.expect_final_weight(resident);
                    first.modulation = modulation[0];
                    first.modulation_generation = modulation_generation[0];
                    second.modulation = modulation[1];
                    second.modulation_generation = modulation_generation[1];
                    first.activated = time[0];
                    first.activated_generation = time_generation[0];
                    second.activated = time[1];
                    second.activated_generation = time_generation[1];
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

        // Conditioning weights upload once; each return to block 1 admits its
        // replicated final weights after block 0 retired them.
        const auto block_bytes = 2 * (64 * 512 + 512 * 64 + 64 * 512);
        check(
            resident.transfers().weight_upload
            == shared_bytes + 4 * block_bytes + 2 * 2 * 2 * 64 * 64 * 2
        );
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
        metadata.available = conditioning_budget;
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
    for (const int unsupported : {0, 1, 2, 3, 4, 5}) {
        const bool native = unsupported != 0;
        write_weights(
            directory,
            false,
            64,
            512,
            {.denoiser = true,
             .modulation_f32 = unsupported == 1,
             .timestep_f32 = unsupported == 2,
             .linear1_f32 = unsupported == 3,
             .final_f32 = unsupported == 4,
             .output_f32 = unsupported == 5}
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
                   == (unsupported >= 4   ? "image_final_standard_precision"
                       : unsupported == 1 ? "image_modulation_standard_precision"
                                          : "image_timestep_standard_precision")
            && !metadata.probes && !metadata.uploaded && metadata.allocations.empty()
        );
        resident.finish();
    }

    for (const char* projection :
         {"img_in",
          "txt_in.in_layer",
          "txt_in.out_layer",
          "attn.to_q",
          "attn.to_k",
          "attn.to_v"}) {
        const bool qkv_weight = std::string_view(projection).starts_with("attn.");
        write_weights(
            directory,
            false,
            64,
            512,
            {.denoiser = true,
             .ingress_f32 = qkv_weight ? "" : projection,
             .qkv_f32 = qkv_weight ? projection : ""}
        );
        denoiser.load(directory, files, config, cancel);
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
        const auto error = denoiser.prepare_feed_forward(
            resident,
            0,
            17,
            qkv_weight ? HiddenBoundary::pre_qkv : HiddenBoundary::pre_attention,
            cancel
        );
        check(
            error
            && error->detail
                   == (qkv_weight ? "image_attention_standard_precision"
                                  : "image_ingress_standard_precision")
            && !metadata.probes && !metadata.uploaded && metadata.allocations.empty()
        );
        resident.finish();
        denoiser.unload();
    }
    for (const char* weight :
         {"img_in.weight",
          "txt_in.text_norm.weight",
          "txt_in.in_layer.weight",
          "txt_in.out_layer.weight",
          "transformer_blocks.0.attn.to_q.weight",
          "transformer_blocks.0.attn.to_k.weight",
          "transformer_blocks.0.attn.to_v.weight"}) {
        for (bool norm_f32 : {false, true}) {
            if (norm_f32 && std::string_view(weight) != "txt_in.text_norm.weight")
                continue;
            write_weights(
                directory, false, 64, 512, {.denoiser = true, .norm_f32 = norm_f32}
            );
            std::uint64_t offset = 0;
            {
                kadan::checkpoint::Shard shard(
                    directory,
                    "weights.safetensors",
                    std::make_shared<kadan::checkpoint::MemoryBudget>(16 * 1024 * 1024)
                );
                offset = shard.tensor_identity(weight).payload_begin;
            }
            {
                std::fstream file(
                    std::string(directory) + "/weights.safetensors",
                    std::ios::in | std::ios::out | std::ios::binary
                );
                file.seekp(std::streamoff(offset));
                if (norm_f32) {
                    file.put(0);
                    file.put(0);
                }
                file.put(char(0x80));
                file.put(char(0x7f));
                check(bool(file));
            }
            denoiser.load(directory, files, config, cancel);
            auto fake = std::make_unique<Device>();
            auto& metadata = *fake;
            metadata.metadata_execution = true;
            auto weight_resources = std::make_shared<kadan::Resources>(
                kadan::Footprint{128 * 1024 * 1024, 32 * 1024 * 1024}
            );
            kadan::MediaSession resident(
                weight_resources,
                activity,
                std::move(fake),
                kadan::Workload::image,
                {{0}},
                cancel
            );
            bool rejected = false;
            try {
                denoiser.prepare_feed_forward(
                    resident,
                    0,
                    17,
                    std::string_view(weight).starts_with("transformer_blocks.")
                        ? HiddenBoundary::pre_qkv
                        : HiddenBoundary::pre_attention,
                    cancel
                );
            } catch (const std::runtime_error& e) {
                rejected = true;
                check(std::string_view(e.what()) == "h3_cuda_nonfinite_weight");
            }
            check(rejected && metadata.allocations.empty() && activity->owners == 0);
            denoiser.unload();
        }
    }

    struct CorruptWeight {
        const char* name;
        std::streamoff bytes_from_end;
    };

    for (const auto weight :
         {CorruptWeight{"modulation.1", 49152},
          CorruptWeight{"time_text_embed.timestep_embedder.linear_2", 57344},
          CorruptWeight{"time_text_embed.timestep_embedder.linear_1", 90112},
          CorruptWeight{"norm_out.linear", 16384},
          CorruptWeight{"proj_out", 8192}}) {
        write_weights(directory, false, 64, 512, {.denoiser = true});
        {
            // Shared weights occupy the tail in write_weights' declared order.
            std::fstream file(
                std::string(directory) + "/weights.safetensors",
                std::ios::in | std::ios::out | std::ios::binary
            );
            file.seekp(-weight.bytes_from_end, std::ios::end);
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
            if (!rejected)
                throw std::runtime_error(
                    std::string("accepted_nonfinite_weight:") + weight.name
                );
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
