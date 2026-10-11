#include <array>
#include <bit>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>

#include "archive_reader.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
// A data-only reader for the verified Whisper torch.save format. GLOBAL/REDUCE
// accept only storage/tensor descriptors and OrderedDict; nothing is executed.
class WhisperPickle {
public:
    explicit WhisperPickle(std::string bytes) : bytes_(std::move(bytes)) {
    }

    Json read() {
        std::size_t operations = 0;
        while (at_ < bytes_.size()) {
            preparation_check(
                ++operations <= 1000000 && stack_.size() <= 100000,
                "Checkpoint pickle bound"
            );
            switch (byte()) {
            case 0x80:
                preparation_check(byte() <= 5, "Unsupported pickle protocol");
                break;
            case 0x95:
                integer(8);
                break;  // Frame length; byte bounds still apply.
            case '}':
                push(Json::object());
                break;
            case ']':
            case ')':
                push(Json::array());
                break;
            case '(':
                marks_.push_back(stack_.size());
                break;
            case 'N':
                push(nullptr);
                break;
            case 0x88:
                push(true);
                break;
            case 0x89:
                push(false);
                break;
            case 'K':
                push(integer(1));
                break;
            case 'M':
                push(integer(2));
                break;
            case 'J':
                push(std::int32_t(integer(4)));
                break;
            case 0x8a: {
                const auto length = byte();
                preparation_check(length <= 8, "Pickle integer bound");
                push(integer(length));
                break;
            }
            case 'X':
                push(text(integer(4)));
                break;
            case 0x8c:
            case 'U':
                push(text(byte()));
                break;
            case 'T':
                push(text(integer(4)));
                break;
            case 'q':
                memo(integer(1));
                break;
            case 'r':
                memo(integer(4));
                break;
            case 0x94:
                memo(memo_.size());
                break;
            case 'h':
                get(integer(1));
                break;
            case 'j':
                get(integer(4));
                break;
            case 't':
                tuple(mark());
                break;
            case 0x85:
                tuple(stack_.size() - 1);
                break;
            case 0x86:
                preparation_check(stack_.size() >= 2, "Invalid pickle tuple");
                tuple(stack_.size() - 2);
                break;
            case 0x87:
                preparation_check(stack_.size() >= 3, "Invalid pickle tuple");
                tuple(stack_.size() - 3);
                break;
            case 'u':
                setitems(mark());
                break;
            case 's':
                preparation_check(stack_.size() >= 3, "Invalid pickle item");
                setitems(stack_.size() - 2);
                break;
            case 'e':
                append(mark());
                break;
            case 'a':
                preparation_check(stack_.size() >= 2, "Invalid pickle append");
                append(stack_.size() - 1);
                break;
            case 'c': {
                const auto module = line(), name = line();
                push({{"global", module + "." + name}});
                break;
            }
            case 0x93: {
                auto name = pop(), module = pop();
                push(
                    {{"global",
                      module->get<std::string>() + "." + name->get<std::string>()}}
                );
                break;
            }
            case 'Q':
                persistent();
                break;
            case 'R':
                reduce();
                break;
            case 'b': {
                const auto attributes = pop();
                preparation_check(
                    !stack_.empty() && attributes->is_object(),
                    "Unsupported pickle attributes"
                );
                // OrderedDict's _metadata describes module versions, not weights.
                for (auto it = attributes->begin(); it != attributes->end(); ++it)
                    preparation_check(
                        it.key() == "_metadata", "Unsupported pickle attribute"
                    );
                break;
            }
            case '.':
                preparation_check(
                    stack_.size() == 1 && marks_.empty() && at_ == bytes_.size(),
                    "Invalid pickle termination"
                );
                return *stack_.back();
            default:
                throw RequestError("Unsupported Whisper checkpoint pickle opcode");
            }
        }
        throw RequestError("Truncated Whisper checkpoint pickle");
    }

private:
    unsigned char byte() {
        preparation_check(at_ < bytes_.size(), "Truncated pickle");
        return static_cast<unsigned char>(bytes_[at_++]);
    }

    std::uint64_t integer(std::size_t size) {
        std::uint64_t result = 0;
        for (std::size_t i = 0; i < size; ++i)
            result |= std::uint64_t(byte()) << (i * 8);
        return result;
    }

    std::string text(std::size_t size) {
        preparation_check(
            size <= 4 * 1024 * 1024 && size <= bytes_.size() - at_,
            "Pickle string bound"
        );
        auto value = bytes_.substr(at_, size);
        at_ += size;
        return value;
    }

    std::string line() {
        const auto end = bytes_.find('\n', at_);
        preparation_check(
            end != std::string::npos && end - at_ <= 256, "Invalid pickle global"
        );
        auto value = text(end - at_);
        ++at_;
        return value;
    }

    void push(Json value) {
        stack_.push_back(std::make_shared<Json>(std::move(value)));
    }

    std::shared_ptr<Json> pop() {
        preparation_check(!stack_.empty(), "Empty pickle stack");
        auto value = stack_.back();
        stack_.pop_back();
        return value;
    }

    std::size_t mark() {
        preparation_check(!marks_.empty(), "Missing pickle mark");
        auto at = marks_.back();
        marks_.pop_back();
        return at;
    }

    void memo(std::size_t index) {
        preparation_check(index < 100000 && !stack_.empty(), "Pickle memo bound");
        memo_[index] = stack_.back();
    }

    void get(std::size_t index) {
        preparation_check(memo_.contains(index), "Invalid pickle memo");
        stack_.push_back(memo_.at(index));
    }

    void tuple(std::size_t start) {
        preparation_check(start <= stack_.size(), "Invalid pickle tuple");
        Json value = Json::array();
        for (std::size_t i = start; i < stack_.size(); ++i)
            value.push_back(*stack_[i]);
        stack_.resize(start);
        push(value);
    }

    void setitems(std::size_t start) {
        preparation_check(
            start > 0 && start <= stack_.size() && (stack_.size() - start) % 2 == 0
                && stack_[start - 1]->is_object(),
            "Invalid pickle dictionary"
        );
        auto& object = *stack_[start - 1];
        for (std::size_t i = start; i < stack_.size(); i += 2)
            object[stack_[i]->get<std::string>()] = *stack_[i + 1];
        stack_.resize(start);
    }

    void append(std::size_t start) {
        preparation_check(
            start > 0 && stack_[start - 1]->is_array(), "Invalid pickle list"
        );
        for (std::size_t i = start; i < stack_.size(); ++i)
            stack_[start - 1]->push_back(*stack_[i]);
        stack_.resize(start);
    }

    void persistent() {
        const auto record = *pop();
        preparation_check(
            record.is_array() && record.size() == 5 && record[0] == "storage",
            "Unsupported pickle persistent object"
        );
        const auto global = record[1].at("global").get<std::string>();
        std::string dtype;
        if (global == "torch.FloatStorage")
            dtype = "F32";
        else if (global == "torch.HalfStorage")
            dtype = "F16";
        else if (global == "torch.BFloat16Storage")
            dtype = "BF16";
        else
            throw RequestError("Unsupported Whisper storage dtype");
        const auto key = record[2].get<std::string>();
        preparation_check(
            !key.empty() && key.size() < 128
                && key.find_first_not_of("0123456789") == std::string::npos,
            "Invalid Whisper storage key"
        );
        push({{"storage", key}, {"dtype", dtype}, {"elements", record[4]}});
    }

    void reduce() {
        const auto args = *pop(), callable = *pop();
        const auto global = callable.at("global").get<std::string>();
        preparation_check(args.is_array(), "Invalid pickle reducer arguments");
        if (global == "collections.OrderedDict") {
            preparation_check(args.empty(), "Unsupported OrderedDict initializer");
            push(Json::object());
        } else if (
            global == "torch._utils._rebuild_tensor_v2"
            || global == "torch._utils._rebuild_tensor"
        ) {
            preparation_check(
                args.size() >= 4 && args.size() <= 7 && args[0].contains("storage"),
                "Invalid tensor descriptor"
            );
            Json tensor = args[0];
            tensor["offset"] = args[1];
            tensor["shape"] = args[2];
            tensor["stride"] = args[3];
            push(tensor);
        } else if (global == "torch._utils._rebuild_parameter") {
            preparation_check(
                args.size() == 3 && args[0].contains("storage"),
                "Invalid parameter descriptor"
            );
            push(args[0]);
        } else
            throw RequestError("Unsupported Whisper pickle global: " + global);
    }

    std::string bytes_;
    std::size_t at_ = 0;
    std::vector<std::shared_ptr<Json>> stack_;
    std::vector<std::size_t> marks_;
    std::unordered_map<std::size_t, std::shared_ptr<Json>> memo_;
};

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
                checkpoint =
                    WhisperPickle(archive.read(4 * 1024 * 1024, cancel)).read();
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
