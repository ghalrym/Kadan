#include <array>
#include <bit>
#include <fstream>
#include <map>
#include <set>

#include "archive_reader.hpp"
#include "checkpoint_pickle.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {

struct Tensor {
    std::string name, storage, dtype;
    std::vector<std::uint64_t> shape, stride;
    std::uint64_t offset = 0, count = 1, storage_count = 0, file_offset = 0;
    unsigned item_size = 0;
};
}  // namespace

void export_whisper(
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
            && fs::file_size(source) <= 8ULL * 1024 * 1024 * 1024,
        "Invalid Whisper source"
    );
    preparation_check(!fs::exists(destination), "Whisper export destination exists");
    const auto modified = fs::last_write_time(source);
    const auto size = fs::file_size(source);
    preparation_check(
        file_digest(source, cancel) == expected, "Whisper source SHA-256 mismatch"
    );
    Json checkpoint;
    std::string prefix;
    {
        ArchiveReader archive(source);
        while (archive.next())
            if (archive.name().ends_with("/data.pkl")) {
                preparation_check(checkpoint.is_null(), "Duplicate checkpoint pickle");
                prefix = archive.name().substr(0, archive.name().size() - 8);
                checkpoint = read_checkpoint_pickle(
                    archive.read(4 * 1024 * 1024, cancel), cancel
                );
            }
    }
    preparation_check(checkpoint.is_object(), "Missing Whisper checkpoint metadata");
    const std::array<const char*, 10> names = {
        "n_mels",
        "n_audio_ctx",
        "n_audio_state",
        "n_audio_head",
        "n_audio_layer",
        "n_vocab",
        "n_text_ctx",
        "n_text_state",
        "n_text_head",
        "n_text_layer"
    };
    const auto& dims = checkpoint.at("dims");
    preparation_check(
        dims.is_object() && dims.size() == names.size(), "Invalid Whisper dimensions"
    );
    for (const auto* name : names)
        preparation_check(
            dims.at(name).is_number_integer() && dims.at(name) > 0
                && dims.at(name) <= 52000,
            "Invalid Whisper dimension"
        );
    const auto& state = checkpoint.at("model_state_dict");
    preparation_check(
        state.is_object() && !state.empty() && state.size() <= 4096,
        "Invalid Whisper state dictionary"
    );
    std::vector<Tensor> tensors;
    Json header = Json::object();
    std::uint64_t output_size = 0;
    for (auto it = state.begin(); it != state.end(); ++it) {
        const auto& value = it.value();
        Tensor tensor;
        tensor.name = it.key();
        tensor.storage = value.at("storage");
        tensor.dtype = value.at("dtype");
        tensor.shape = value.at("shape").get<std::vector<std::uint64_t>>();
        tensor.stride = value.at("stride").get<std::vector<std::uint64_t>>();
        tensor.offset = value.at("offset");
        tensor.storage_count = value.at("elements");
        tensor.item_size = tensor.dtype == "F32" ? 4 : 2;
        preparation_check(
            tensor.name.size() <= 256 && !tensor.shape.empty()
                && tensor.shape.size() <= 3
                && tensor.stride.size() == tensor.shape.size(),
            "Invalid Whisper tensor shape"
        );
        preparation_check(
            tensor.offset < tensor.storage_count, "Whisper tensor offset bound"
        );
        std::uint64_t maximum = tensor.offset;
        for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
            preparation_check(
                tensor.shape[i] > 0
                    && tensor.shape[i]
                           <= 512 * 1024 * 1024 / tensor.item_size / tensor.count,
                "Whisper tensor size bound"
            );
            tensor.count *= tensor.shape[i];
            preparation_check(
                tensor.stride[i] <= 512 * 1024 * 1024 / tensor.item_size
                    && tensor.shape[i] <= 512 * 1024 * 1024 / tensor.item_size,
                "Whisper tensor stride bound"
            );
            maximum += (tensor.shape[i] - 1) * tensor.stride[i];
        }
        preparation_check(
            maximum < tensor.storage_count
                && tensor.storage_count <= 512 * 1024 * 1024 / tensor.item_size,
            "Whisper tensor storage bound"
        );
        tensor.file_offset = output_size;
        output_size += tensor.count * tensor.item_size;
        preparation_check(
            output_size <= 8ULL * 1024 * 1024 * 1024, "Whisper checkpoint size bound"
        );
        header[tensor.name] = {
            {"dtype", tensor.dtype},
            {"shape", tensor.shape},
            {"data_offsets", {tensor.file_offset, output_size}}
        };
        tensors.push_back(std::move(tensor));
    }
    const auto serialized = header.dump();
    preparation_check(
        serialized.size() <= 4 * 1024 * 1024, "Whisper header size bound"
    );
    fs::create_directories(destination.parent_path());
    const auto stage = fs::path(destination.string() + ".partial");
    fs::remove_all(stage);
    fs::create_directory(stage);
    try {
        preparation_check(
            fs::space(stage).available >= output_size + 64 * 1024 * 1024,
            "Insufficient space for Whisper export"
        );
        std::ofstream output(stage / "model.safetensors", std::ios::binary);
        const std::uint64_t header_size = serialized.size();
        static_assert(std::endian::native == std::endian::little);
        output.write(reinterpret_cast<const char*>(&header_size), 8);
        output << serialized;
        std::set<std::string> written;
        ArchiveReader archive(source);
        while (archive.next()) {
            preparation_cancel(cancel);
            const auto name = archive.name();
            if (!name.starts_with(prefix + "data/"))
                continue;
            const auto key = name.substr(prefix.size() + 5);
            std::vector<const Tensor*> matching;
            for (const auto& tensor : tensors)
                if (tensor.storage == key)
                    matching.push_back(&tensor);
            if (matching.empty())
                continue;
            const auto bytes = archive.read(512 * 1024 * 1024, cancel);
            for (const auto* tensor : matching) {
                preparation_check(
                    bytes.size() == tensor->storage_count * tensor->item_size
                        && written.insert(tensor->name).second,
                    "Whisper storage size mismatch"
                );
                output.seekp(8 + header_size + tensor->file_offset);
                bool contiguous = true;
                std::uint64_t stride = 1;
                for (std::size_t d = tensor->shape.size(); d > 0; --d) {
                    if (tensor->shape[d - 1] > 1 && tensor->stride[d - 1] != stride)
                        contiguous = false;
                    stride *= tensor->shape[d - 1];
                }
                if (contiguous)
                    output.write(
                        bytes.data() + tensor->offset * tensor->item_size,
                        tensor->count * tensor->item_size
                    );
                else
                    for (std::uint64_t i = 0; i < tensor->count; ++i) {
                        if (i % 262144 == 0)
                            preparation_cancel(cancel);
                        auto remaining = i, at = tensor->offset;
                        for (std::size_t d = tensor->shape.size(); d > 0; --d) {
                            at += (remaining % tensor->shape[d - 1])
                                  * tensor->stride[d - 1];
                            remaining /= tensor->shape[d - 1];
                        }
                        output.write(
                            bytes.data() + at * tensor->item_size, tensor->item_size
                        );
                    }
            }
        }
        output.close();
        preparation_check(
            !output.fail() && written.size() == tensors.size(),
            "Whisper export is incomplete"
        );
        std::ofstream dimensions(stage / "dimensions.txt");
        for (const auto* name : names)
            dimensions << dims.at(name).get<unsigned>() << ' ';
        dimensions << '\n';
        dimensions.close();
        preparation_check(!dimensions.fail(), "Cannot write Whisper dimensions");
        write_document(
            stage / "export.json",
            {{"source_sha256", expected},
             {"source_bytes", size},
             {"tensors", tensors.size()},
             {"dimensions", dims},
             {"model_execution", false}}
        );
        preparation_cancel(cancel);
        preparation_check(
            fs::file_size(source) == size && fs::last_write_time(source) == modified,
            "Whisper source changed during conversion"
        );
        fs::rename(stage, destination);
    } catch (...) {
        fs::remove_all(stage);
        throw;
    }
}
}  // namespace kadan::serving
