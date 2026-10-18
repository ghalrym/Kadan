#include "codec_export.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>
#include <set>

#include "archive_reader.hpp"
#include "kadan/drama_codec.hpp"

namespace kadan::serving {
namespace {
constexpr std::uint64_t storage_limit = 256ULL * 1024 * 1024;
constexpr std::uint64_t checkpoint_limit = 4ULL * 1024 * 1024 * 1024;

struct CodecStorage {
    std::string key;
    std::vector<std::uint64_t> shape, stride;
    std::uint64_t offset, elements, count = 1;

    explicit CodecStorage(const Json& value) {
        preparation_check(
            value.at("dtype") == "F32", "Drama codec requires FP32 source weights"
        );
        key = value.at("storage").get<std::string>();
        shape = value.at("shape").get<std::vector<std::uint64_t>>();
        stride = value.at("stride").get<std::vector<std::uint64_t>>();
        offset = value.at("offset").get<std::uint64_t>();
        elements = value.at("elements").get<std::uint64_t>();
        preparation_check(
            !shape.empty() && shape.size() <= 4 && stride.size() == shape.size(),
            "Invalid Drama codec tensor rank"
        );
        preparation_check(
            elements > 0 && elements <= storage_limit / 4 && offset < elements,
            "Drama codec storage bound"
        );
        std::uint64_t maximum = offset;
        for (std::size_t i = 0; i < shape.size(); ++i) {
            preparation_check(
                shape[i] > 0 && shape[i] <= storage_limit / 4 / count,
                "Drama codec tensor size bound"
            );
            count *= shape[i];
            preparation_check(
                stride[i] <= storage_limit / 4
                    && (shape[i] - 1) * stride[i] < elements - maximum,
                "Drama codec tensor stride bound"
            );
            maximum += (shape[i] - 1) * stride[i];
        }
    }

    float at(std::string_view bytes, std::uint64_t index) const {
        auto position = offset;
        for (std::size_t d = shape.size(); d > 0; --d) {
            position += (index % shape[d - 1]) * stride[d - 1];
            index /= shape[d - 1];
        }
        float value;
        std::memcpy(&value, bytes.data() + position * 4, 4);
        preparation_check(std::isfinite(value), "Nonfinite Drama codec weight");
        return value;
    }
};

struct CodecExportTensor {
    std::string name;
    CodecStorage values;
    std::optional<CodecStorage> gains;
    std::uint64_t output_offset;
    std::vector<float> gain_values;
    bool written = false;
};

std::vector<CodecExportTensor> codec_tensors(const Json& state) {
    preparation_check(
        state.is_object() && !state.empty() && state.size() <= 16384,
        "Invalid Drama codec state dictionary"
    );
    std::vector<CodecExportTensor> tensors;
    std::uint64_t offset = 0;
    for (const auto& spec : drama::codec_tensor_specs()) {
        std::string values_name = spec.name, gains_name;
        if (spec.name.ends_with(".weight")) {
            const auto prefix = spec.name.substr(0, spec.name.size() - 7);
            const auto legacy_v = prefix + ".weight_v";
            const auto param_v = prefix + ".parametrizations.weight.original1";
            const auto legacy_g = prefix + ".weight_g";
            const auto param_g = prefix + ".parametrizations.weight.original0";
            const bool legacy = state.contains(legacy_v) || state.contains(legacy_g);
            const bool parametrized =
                state.contains(param_v) || state.contains(param_g);
            preparation_check(
                unsigned(state.contains(spec.name)) + unsigned(legacy)
                        + unsigned(parametrized)
                    == 1,
                "Missing or ambiguous Drama codec weight: " + spec.name
            );
            if (legacy) {
                values_name = legacy_v;
                gains_name = legacy_g;
            }
            if (parametrized) {
                values_name = param_v;
                gains_name = param_g;
            }
        }
        preparation_check(
            state.contains(values_name), "Missing Drama codec tensor: " + values_name
        );
        CodecExportTensor tensor{
            spec.name, CodecStorage(state.at(values_name)), {}, offset, {}, false
        };
        preparation_check(
            tensor.values.shape == spec.shape && tensor.values.count == spec.count,
            "Drama codec tensor shape mismatch: " + spec.name
        );
        if (!gains_name.empty()) {
            preparation_check(
                state.contains(gains_name),
                "Missing Drama codec weight norm gain: " + gains_name
            );
            tensor.gains.emplace(state.at(gains_name));
            auto gain_shape = spec.shape;
            std::fill(gain_shape.begin() + 1, gain_shape.end(), 1);
            preparation_check(
                tensor.gains->shape == gain_shape,
                "Drama codec weight norm requires dim 0"
            );
        }
        offset += spec.count * 4;
        preparation_check(
            offset <= checkpoint_limit, "Drama codec checkpoint size bound"
        );
        tensors.push_back(std::move(tensor));
    }
    return tensors;
}

Json codec_header(std::uint64_t& output_size) {
    Json header = Json::object();
    output_size = 0;
    for (const auto& spec : drama::codec_tensor_specs()) {
        const auto start = output_size;
        output_size += spec.count * 4;
        header[spec.name] = {
            {"dtype", "F32"},
            {"shape", spec.shape},
            {"data_offsets", {start, output_size}}
        };
    }
    return header;
}

void validate_cached_export(
    const fs::path& destination,
    const std::string& expected,
    std::uint64_t source_size,
    fs::file_time_type modified
) {
    preparation_check(
        !fs::is_symlink(destination) && fs::is_directory(destination),
        "Invalid Drama codec export directory"
    );
    const auto metadata = read_document(destination / "export.json");
    preparation_check(
        metadata.at("codec_export_version") == 1
            && metadata.at("source_sha256") == expected
            && metadata.at("source_bytes") == source_size
            && metadata.at("source_modified") == modified.time_since_epoch().count(),
        "Drama codec export source identity changed; remove the stale codec export "
        "before retrying"
    );
    const auto path = destination / "model.safetensors";
    preparation_check(
        !fs::is_symlink(path) && fs::is_regular_file(path),
        "Missing Drama codec exported weights"
    );
    std::ifstream input(path, std::ios::binary);
    std::uint64_t header_size = 0, output_size = 0;
    input.read(reinterpret_cast<char*>(&header_size), 8);
    preparation_check(
        input.good() && header_size > 0 && header_size <= 4 * 1024 * 1024,
        "Drama codec export header bound"
    );
    std::string serialized(header_size, '\0');
    input.read(serialized.data(), header_size);
    preparation_check(
        input.good() && Json::parse(serialized) == codec_header(output_size)
            && fs::file_size(path) == 8 + header_size + output_size,
        "Drama codec export schema or size changed; remove the stale codec export "
        "before retrying"
    );
}
}  // namespace

void fold_codec_weight_norm(std::span<float> row, float gain) {
    preparation_check(
        !row.empty() && std::isfinite(gain), "Invalid Drama codec weight norm gain"
    );
    double squared = 0;
    for (const auto value : row) {
        preparation_check(
            std::isfinite(value), "Nonfinite Drama codec weight norm value"
        );
        squared += double(value) * value;
    }
    preparation_check(
        squared > 0 && std::isfinite(squared),
        "Invalid Drama codec weight norm magnitude"
    );
    const double scale = double(gain) / std::sqrt(squared);
    for (auto& value : row) {
        value = float(double(value) * scale);
        preparation_check(
            std::isfinite(value), "Drama codec normalized weight overflow"
        );
    }
}

void export_drama_codec(
    const fs::path& source,
    const std::string& expected,
    const fs::path& destination,
    const std::atomic_bool& cancel
) {
    preparation_cancel(cancel);
    preparation_check(
        expected.size() == 64
            && expected.find_first_not_of("0123456789abcdef") == std::string::npos,
        "Expected SHA-256 is required"
    );
    preparation_check(
        !fs::is_symlink(source) && fs::is_regular_file(source)
            && fs::file_size(source) <= checkpoint_limit,
        "Invalid Drama codec source"
    );
    const auto modified = fs::last_write_time(source);
    const auto size = fs::file_size(source);
    if (fs::exists(destination) || fs::is_symlink(destination)) {
        validate_cached_export(destination, expected, size, modified);
        return;
    }
    preparation_check(
        file_digest(source, cancel) == expected, "Drama codec source SHA-256 mismatch"
    );
    Json state;
    std::string prefix;
    {
        ArchiveReader archive(source);
        while (archive.next()) {
            preparation_cancel(cancel);
            if (archive.name().ends_with("/data.pkl")) {
                preparation_check(state.is_null(), "Duplicate Drama codec pickle");
                prefix = archive.name().substr(0, archive.name().size() - 8);
                state = read_checkpoint_pickle(
                    archive.read(4 * 1024 * 1024, cancel), cancel
                );
            }
        }
    }
    auto tensors = codec_tensors(state);
    state = nullptr;
    std::uint64_t output_size = 0;
    const auto header = codec_header(output_size);
    auto serialized = header.dump();
    serialized.append((8 - serialized.size() % 8) % 8, ' ');
    preparation_check(
        serialized.size() <= 4 * 1024 * 1024, "Drama codec header size bound"
    );
    fs::create_directories(destination.parent_path());
    const fs::path stage = destination.string() + ".partial";
    // Never follow or remove a preexisting partial path owned by another operation.
    preparation_check(
        fs::create_directory(stage), "Drama codec export partial destination exists"
    );
    try {
        preparation_check(
            fs::space(stage).available >= output_size + 64 * 1024 * 1024,
            "Insufficient space for Drama codec export"
        );
        std::fstream output(
            stage / "model.safetensors",
            std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc
        );
        static_assert(std::endian::native == std::endian::little);
        const std::uint64_t header_size = serialized.size();
        output.write(reinterpret_cast<const char*>(&header_size), 8);
        output << serialized;
        std::set<std::string> storages;
        ArchiveReader archive(source);
        while (archive.next()) {
            preparation_cancel(cancel);
            const auto name = archive.name();
            if (name == prefix + "byteorder") {
                preparation_check(
                    archive.read(16, cancel) == "little",
                    "Drama codec requires little-endian source storage"
                );
                continue;
            }
            if (!name.starts_with(prefix + "data/"))
                continue;
            const auto key = name.substr(prefix.size() + 5);
            bool needed = false;
            for (const auto& tensor : tensors)
                needed |= tensor.values.key == key
                          || (tensor.gains && tensor.gains->key == key);
            if (!needed)
                continue;
            preparation_check(
                storages.insert(key).second, "Duplicate Drama codec storage"
            );
            const auto bytes = archive.read(storage_limit, cancel);
            for (auto& tensor : tensors) {
                if (tensor.gains && tensor.gains->key == key) {
                    const auto& gains = *tensor.gains;
                    preparation_check(
                        bytes.size() == gains.elements * 4,
                        "Drama codec gain storage size mismatch"
                    );
                    tensor.gain_values.reserve(gains.count);
                    for (std::uint64_t i = 0; i < gains.count; ++i)
                        tensor.gain_values.push_back(gains.at(bytes, i));
                }
                if (tensor.values.key != key)
                    continue;
                const auto& values = tensor.values;
                preparation_check(
                    bytes.size() == values.elements * 4,
                    "Drama codec storage size mismatch"
                );
                output.seekp(8 + header_size + tensor.output_offset);
                std::array<float, 16384> block;
                for (std::uint64_t i = 0; i < values.count;) {
                    preparation_cancel(cancel);
                    const auto count =
                        std::min<std::uint64_t>(block.size(), values.count - i);
                    for (std::uint64_t j = 0; j < count; ++j)
                        block[j] = values.at(bytes, i + j);
                    output.write(
                        reinterpret_cast<const char*>(block.data()), count * 4
                    );
                    preparation_check(
                        output.good(), "Cannot write Drama codec weights"
                    );
                    i += count;
                }
                tensor.written = true;
            }
        }
        for (const auto& tensor : tensors) {
            preparation_check(
                tensor.written, "Missing Drama codec storage: " + tensor.name
            );
            if (!tensor.gains)
                continue;
            preparation_check(
                tensor.gain_values.size() == tensor.values.shape[0],
                "Missing Drama codec weight norm gains"
            );
            const auto width = tensor.values.count / tensor.values.shape[0];
            std::vector<float> row(width);
            for (std::size_t i = 0; i < tensor.gain_values.size(); ++i) {
                preparation_cancel(cancel);
                const auto position =
                    8 + header_size + tensor.output_offset + i * width * 4;
                output.seekg(position);
                output.read(reinterpret_cast<char*>(row.data()), width * 4);
                preparation_check(
                    output.good(), "Cannot read Drama codec weight norm values"
                );
                fold_codec_weight_norm(row, tensor.gain_values[i]);
                output.seekp(position);
                output.write(reinterpret_cast<const char*>(row.data()), width * 4);
                preparation_check(
                    output.good(), "Cannot write Drama codec normalized weights"
                );
            }
        }
        output.close();
        preparation_check(!output.fail(), "Drama codec export is incomplete");
        write_document(
            stage / "export.json",
            {{"codec_export_version", 1},
             {"source_modified", modified.time_since_epoch().count()},
             {"source_sha256", expected},
             {"source_bytes", size},
             {"tensors", tensors.size()},
             {"weight_norm_dimension", 0},
             {"model_execution", false}}
        );
        preparation_cancel(cancel);
        preparation_check(
            fs::file_size(source) == size && fs::last_write_time(source) == modified,
            "Drama codec source changed during conversion"
        );
        fs::rename(stage, destination);
    } catch (...) {
        fs::remove_all(stage);
        throw;
    }
}
}  // namespace kadan::serving
