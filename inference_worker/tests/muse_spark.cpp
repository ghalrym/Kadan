#include "kadan/muse_spark.hpp"

#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <ggml.h>
#include <gguf.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;

struct Fixture {
    fs::path root;
    Json source;

    Fixture() {
        const auto path =
            fs::path(__FILE__).parent_path() / "fixtures" / "muse_spark_metadata.json";
        std::ifstream input(path);
        source = Json::parse(input);
        assert(source.at("revision") == "b7521af78bddb86c0de111e48fe93ff549711de9");
        std::string pattern = "/tmp/kadan-muse-metadata-XXXXXX";
        const auto* directory = mkdtemp(pattern.data());
        assert(directory);
        root = directory;
        write("config.json", source.at("config"));
        write("generation_config.json", {{"eos_token_id", {200001, 200008}}});
        Json index = {{"weight_map", Json::object()}};
        std::map<std::string, Json> headers;
        for (auto it = source.at("tensors").begin(); it != source.at("tensors").end();
             ++it) {
            auto tensor = it.value();
            const auto shard = tensor.at("shard").get<std::string>();
            tensor.erase("shard");
            headers[shard][it.key()] = tensor;
            index["weight_map"][it.key()] = shard;
        }
        write("model.safetensors.index.json", index);
        for (const auto& [shard, header] : headers) {
            const auto text = header.dump();
            std::ofstream output(root / shard, std::ios::binary);
            for (unsigned byte = 0; byte < 8; ++byte)
                output.put(char((std::uint64_t(text.size()) >> (8 * byte)) & 255));
            output << text;
            output.close();
            std::uint64_t end = 0;
            for (const auto& tensor : header)
                end = std::max(end, tensor.at("data_offsets")[1].get<std::uint64_t>());
            // Sparse holes represent unread payload spans. No weights are stored
            // or executed; only the published tensor headers are under test.
            fs::resize_file(root / shard, 8 + text.size() + end);
        }
    }

    ~Fixture() {
        fs::remove_all(root);
    }

    void write(const std::string& name, const Json& value) {
        std::ofstream(root / name) << value;
    }
};

template <class F> void rejects(F action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::exception&) {
        rejected = true;
    }
    assert(rejected);
}

struct GgufFixture {
    fs::path root;
    Json source;

    GgufFixture() {
        const auto path = fs::path(__FILE__).parent_path() / "fixtures"
                          / "muse_spark_gguf_metadata.json";
        std::ifstream input(path);
        source = Json::parse(input);
        assert(source.at("revision") == "19d0b1b3a1b1974a48439e11e0639c0dad164fc8");
        std::string pattern = "/tmp/kadan-muse-gguf-metadata-XXXXXX";
        const auto* directory = mkdtemp(pattern.data());
        assert(directory);
        root = directory;
        write(source);
    }

    ~GgufFixture() {
        fs::remove_all(root);
    }

    fs::path file(unsigned split) const {
        std::ostringstream name;
        name << "Muse-Spark-Open-816B-A42B-UD-Q3_K_XL-" << std::setfill('0')
             << std::setw(5) << split + 1 << "-of-00010.gguf";
        return root / name.str();
    }

    void write(const Json& description) const {
        std::ofstream(root / "config.json") << description.at("config");
        std::uint32_t total = 0;
        for (const auto& family : description.at("families"))
            total += family.at("members").size();
        const auto splits = description.at("split_count").get<std::uint32_t>();
        assert(splits == 10);
        for (unsigned split = 0; split < splits; ++split) {
            std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
                gguf_init_empty(), gguf_free
            );
            assert(metadata);
            for (auto it = description.at("metadata").begin();
                 it != description.at("metadata").end();
                 ++it) {
                const auto& value = it.value();
                const auto* key = it.key().c_str();
                if (value.is_string())
                    gguf_set_val_str(
                        metadata.get(), key, value.get<std::string>().c_str()
                    );
                else if (value.is_boolean())
                    gguf_set_val_bool(metadata.get(), key, value.get<bool>());
                else if (value.is_number_float())
                    gguf_set_val_f32(metadata.get(), key, value.get<float>());
                else if (value.is_number_integer())
                    gguf_set_val_u32(metadata.get(), key, value.get<std::uint32_t>());
                else {
                    assert(value.is_array());
                    auto array = std::make_unique<bool[]>(value.size());
                    for (std::size_t i = 0; i < value.size(); ++i)
                        array[i] = value[i].get<bool>();
                    gguf_set_arr_data(
                        metadata.get(), key, GGUF_TYPE_BOOL, array.get(), value.size()
                    );
                }
            }
            gguf_set_val_u32(metadata.get(), "split.no", split);
            gguf_set_val_u32(metadata.get(), "split.count", splits);
            gguf_set_val_u32(metadata.get(), "split.tensors.count", total);
            // This context contains descriptors only. It never allocates tensor
            // buffers, executes a graph, or reads any model-weight bytes.
            std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(
                ggml_init(
                    {(std::size_t(total) + 1) * ggml_tensor_overhead(), nullptr, true}
                ),
                ggml_free
            );
            assert(tensors);
            for (auto it = description.at("families").begin();
                 it != description.at("families").end();
                 ++it) {
                const auto& family = it.value();
                std::array<std::int64_t, 3> shape{};
                const auto rank = family.at("shape").size();
                for (std::size_t i = 0; i < rank; ++i)
                    shape[i] = family.at("shape")[i].get<std::int64_t>();
                for (const auto& member : family.at("members")) {
                    if (member[1] != split)
                        continue;
                    auto name = it.key();
                    if (const auto wildcard = name.find('*');
                        wildcard != std::string::npos)
                        name.replace(
                            wildcard, 1, std::to_string(member[0].get<unsigned>())
                        );
                    auto* tensor = ggml_new_tensor(
                        tensors.get(),
                        static_cast<ggml_type>(family.at("type").get<unsigned>()),
                        int(rank),
                        shape.data()
                    );
                    assert(tensor && tensor->data == nullptr);
                    ggml_set_name(tensor, name.c_str());
                    gguf_add_tensor(metadata.get(), tensor);
                }
            }
            const auto path = file(split);
            assert(gguf_write_to_file(metadata.get(), path.c_str(), true));
            auto end = std::uint64_t(gguf_get_meta_size(metadata.get()));
            const auto first = end;
            for (std::int64_t i = 0; i < gguf_get_n_tensors(metadata.get()); ++i)
                end = std::max(
                    end,
                    first + gguf_get_tensor_offset(metadata.get(), i)
                        + gguf_get_tensor_size(metadata.get(), i)
                );
            fs::resize_file(path, end);
        }
    }
};

void gguf_plan_units() {
    using namespace kadan::muse_spark;
    GgufFixture fixture;
    const auto root = fixture.root.string();
    const auto layout = plan(root.c_str(), 8192);
    assert(layout.context == 8192 && layout.tensors == 1429);
    assert(layout.packed == 401470321664ULL);
    assert(layout.host < 4ULL * 1024 * 1024 * 1024);
    assert(layout.execution_weights > layout.packed);
    // Both formats reserve the same bounded state; neither admits the complete
    // checkpoint into RAM or VRAM before it can execute a selected expert.
    assert(layout.host == (640ULL << 20) + state_bytes(8192));
    std::atomic_bool cancelled{true};
    rejects([&] { plan(root.c_str(), 8192, &cancelled); });

    auto changed = fixture.source;
    changed["metadata"]["muse-spark.attention.head_count"] = 64;
    fixture.write(changed);
    rejects([&] { plan(root.c_str(), 8192); });

    changed = fixture.source;
    changed["families"]["blk.*.attn_q.weight"]["shape"][1] = 8191;
    fixture.write(changed);
    rejects([&] { plan(root.c_str(), 8192); });

    changed = fixture.source;
    changed["families"]["blk.*.attn_q.weight"]["type"] = 0;
    fixture.write(changed);
    rejects([&] { plan(root.c_str(), 8192); });

    changed = fixture.source;
    changed["families"]["blk.*.attn_q.weight"]["members"].erase(0);
    fixture.write(changed);
    rejects([&] { plan(root.c_str(), 8192); });

    fixture.write(fixture.source);
    fs::rename(fixture.file(9), fixture.root / "missing-split");
    rejects([&] { plan(root.c_str(), 8192); });
    fs::rename(fixture.root / "missing-split", fixture.file(9));
    auto config = fixture.source.at("config");
    config["dtype"] = "bfloat16";
    std::ofstream(fixture.root / "config.json") << config;
    auto budget = std::make_shared<kadan::checkpoint::MemoryBudget>(4 << 20);
    rejects([&] { validate_config(root.c_str(), budget); });
    config = fixture.source.at("config");
    config["quantization_config"] = {
        {"bits", 3}, {"mode", "affine"}, {"group_size", 64}
    };
    std::ofstream(fixture.root / "config.json") << config;
    rejects([&] { validate_config(root.c_str(), budget); });
}
}  // namespace

int main() {
    using namespace kadan::muse_spark;
    gguf_plan_units();
    Fixture fixture;
    const auto root = fixture.root.string();
    auto budget = std::make_shared<kadan::checkpoint::MemoryBudget>(4 << 20);
    validate_config(root.c_str(), budget);
    assert(budget->used() == 0);
    const auto layout = plan(root.c_str(), 8192);
    assert(layout.context == 8192);
    assert(layout.packed == 359213905920ULL);
    assert(layout.tensors == 2425);
    // Resident working memory excludes the hundreds of GiB of disk weights.
    assert(layout.host < 4ULL * 1024 * 1024 * 1024);
    assert(state_bytes(16384) > state_bytes(8192));
    rejects([&] { state_bytes(0); });
    rejects([&] { state_bytes(1048577); });
    std::atomic_bool cancelled{true};
    rejects([&] { plan(root.c_str(), 8192, &cancelled); });
    auto config = fixture.source.at("config");
    config["quantization_config"]["bits"] = 2;
    fixture.write("config.json", config);
    rejects([&] { validate_config(root.c_str(), budget); });
    config = fixture.source.at("config");
    config["text_config"]["layer_types"][1] = "sliding_attention";
    fixture.write("config.json", config);
    rejects([&] { validate_config(root.c_str(), budget); });
    fixture.write("config.json", fixture.source.at("config"));
    fixture.write("generation_config.json", {{"eos_token_id", {200001}}});
    rejects([&] { plan(root.c_str(), 8192); });
    fixture.write("generation_config.json", {{"eos_token_id", {200001, 200008}}});
    fixture.write("model.safetensors.index.json", {{"weight_map", Json::object()}});
    rejects([&] { plan(root.c_str(), 8192); });
    std::cout << "Muse Spark checkpoint metadata and disk-streaming admission units "
                 "passed; no inference\n";
}
