#include <algorithm>
#include <array>
#include <chrono>
#include <curl/curl.h>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <openssl/evp.h>
#include <regex>
#include <set>
#include <sys/file.h>
#include <thread>
#include <unistd.h>

#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
struct Transfer {
    std::function<void(std::size_t)> progress;
    std::string bytes;
    std::ofstream* output = nullptr;
    EVP_MD_CTX* digest = nullptr;
    std::uint64_t received = 0, maximum = 32 * 1024 * 1024;
};

std::size_t
receive(char* bytes, std::size_t size, std::size_t count, void* context) noexcept {
    auto& target = *static_cast<Transfer*>(context);
    try {
        if (size && count > SIZE_MAX / size)
            return 0;
        const auto length = size * count;
        if (length > target.maximum - target.received)
            return 0;
        if (target.output) {
            target.output->write(bytes, length);
            if (!*target.output || EVP_DigestUpdate(target.digest, bytes, length) != 1)
                return 0;
        } else
            target.bytes.append(bytes, length);
        target.received += length;
        if (target.progress)
            target.progress(length);
        return length;
    } catch (...) {
        return 0;
    }
}

int interrupted(void* context, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    return static_cast<const std::atomic_bool*>(context)->load() ? 1 : 0;
}

curl_off_t transfer(
    const std::string& url,
    Transfer& target,
    const std::atomic_bool& cancel,
    bool head = false
) {
    preparation_cancel(cancel);
    static const auto initialized = curl_global_init(CURL_GLOBAL_DEFAULT);
    preparation_check(initialized == CURLE_OK, "Network initialization failed");
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(
        curl_easy_init(), curl_easy_cleanup
    );
    preparation_check(bool(curl), "Network initialization failed");
    preparation_check(url.starts_with("https://"), "Checkpoint URL must use HTTPS");
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "https");
    curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "https");
    curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_MAXREDIRS, 8L);
    curl_easy_setopt(curl.get(), CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl.get(), CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_LOW_SPEED_TIME, 30L);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, receive);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &target);
    curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, interrupted);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &cancel);
    curl_easy_setopt(curl.get(), CURLOPT_NOBODY, head ? 1L : 0L);
    const auto status = curl_easy_perform(curl.get());
    preparation_cancel(cancel);
    preparation_check(
        status == CURLE_OK,
        "Checkpoint download failed; check network and storage.",
        503
    );
    curl_off_t length = -1;
    curl_easy_getinfo(curl.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &length);
    return length;
}

std::int64_t fetch(
    const ModelStore::Fetch& source,
    const std::string& url,
    Transfer& target,
    const std::atomic_bool& cancel,
    bool head = false
) {
    if (!source)
        return transfer(url, target, cancel, head);
    preparation_cancel(cancel);
    const auto result = source(
        url,
        [&](std::string_view chunk) {
            preparation_cancel(cancel);
            preparation_check(
                receive(const_cast<char*>(chunk.data()), 1, chunk.size(), &target)
                    == chunk.size(),
                "Checkpoint transfer exceeds its bound or could not be written"
            );
        },
        cancel,
        head
    );
    preparation_cancel(cancel);
    return result;
}

struct StoreLock {
    int descriptor;

    StoreLock(const fs::path& path, const std::atomic_bool& cancel)
        : descriptor(
              open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600)
          ) {
        preparation_check(descriptor >= 0, "Cannot lock model storage", 503);
        try {
            while (flock(descriptor, LOCK_EX | LOCK_NB) != 0) {
                preparation_check(
                    errno == EWOULDBLOCK || errno == EAGAIN,
                    "Model storage lock failed",
                    503
                );
                preparation_cancel(cancel);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        } catch (...) {
            close(descriptor);
            throw;
        }
    }

    ~StoreLock() {
        close(descriptor);
    }

    StoreLock(const StoreLock&) = delete;
};

bool safe_name(const std::string& name) {
    fs::path path(name);
    if (name.empty() || path.is_absolute() || name.find('\\') != std::string::npos
        || path.lexically_normal().generic_string() != name)
        return false;
    return std::none_of(path.begin(), path.end(), [](const auto& part) {
        return part == "..";
    });
}

bool hex_digest(const std::string& value, std::size_t size) {
    return value.size() == size
           && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}

void download(
    const std::string& url,
    const fs::path& path,
    const Json& item,
    const std::atomic_bool& cancel,
    const ModelStore::Fetch& source,
    const std::function<void(std::size_t)>& progress
) {
    const auto size = item.at("size").get<std::uint64_t>();
    const bool sha256 = item.at("algorithm") == "sha256";
    preparation_check(
        sha256 || item.at("algorithm") == "git-sha1", "Unknown digest algorithm"
    );
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest(
        EVP_MD_CTX_new(), EVP_MD_CTX_free
    );
    preparation_check(
        digest
            && EVP_DigestInit_ex(
                   digest.get(), sha256 ? EVP_sha256() : EVP_sha1(), nullptr
               ) == 1,
        "Digest initialization failed"
    );
    if (!sha256) {
        const auto header = "blob " + std::to_string(size) + std::string(1, '\0');
        preparation_check(
            EVP_DigestUpdate(digest.get(), header.data(), header.size()) == 1,
            "Digest update failed"
        );
    }
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    preparation_check(bool(output), "Cannot write checkpoint", 503);
    Transfer target;
    target.output = &output;
    target.progress = progress;
    target.digest = digest.get();
    target.maximum = size;
    fetch(source, url, target, cancel);
    output.close();
    preparation_check(
        !output.fail() && target.received == size, "Checkpoint size mismatch"
    );
    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
    unsigned count = 0;
    preparation_check(
        EVP_DigestFinal_ex(digest.get(), bytes.data(), &count) == 1,
        "Digest finalization failed"
    );
    constexpr char hex[] = "0123456789abcdef";
    std::string actual;
    for (unsigned i = 0; i < count; ++i) {
        actual += hex[bytes[i] >> 4];
        actual += hex[bytes[i] & 15];
    }
    preparation_check(
        actual == item.at("digest").get<std::string>(),
        "Checkpoint integrity verification failed: " + path.filename().string()
    );
}

void atomic_document(const fs::path& destination, const Json& value) {
    const auto temporary = destination.string() + ".worker.tmp";
    write_document(temporary, value);
    fs::rename(temporary, destination);
}
}  // namespace

ModelStore::ModelStore(
    fs::path root, fs::path data, Fetch fetch, DownloadProgress progress
)
    : root_(std::move(root)), data_(std::move(data)),
      catalog_(read_document(data_ / "catalog.json")), fetch_(std::move(fetch)),
      progress_(std::move(progress)) {
}

const Json& ModelStore::entry(const std::string& model) const {
    for (const auto& candidate : catalog_.at("models"))
        if (candidate.at("id").get<std::string>() == model)
            return candidate;
    throw RequestError("Unknown catalog model: " + model, 422);
}

bool ModelStore::allowed(const std::string& name, const Json& model) const {
    if (!safe_name(name))
        return false;
    const auto layout = model.at("layout").get<std::string>();
    const auto& required = model.at("required_files");
    const auto contains = [](const Json& values, const std::string& value) {
        return std::find(values.begin(), values.end(), Json(value)) != values.end();
    };
    if (layout == "composite")
        return contains(required, name);
    if (layout == "single_file")
        return model.at("asset_name").get<std::string>() == name
               && fs::path(name).parent_path().empty();
    auto basename = name;
    if (layout == "components") {
        if (contains(required, name) || name == "LICENSE" || name == "LICENSE.txt"
            || name == "NOTICE" || name == "README.md")
            return true;
        auto parent = fs::path(name).parent_path().generic_string();
        if (parent.empty())
            parent = ".";
        if (!contains(model.at("component_paths"), parent))
            return false;
        basename = fs::path(name).filename().string();
    } else if (fs::path(name).has_parent_path())
        return false;
    if (contains(catalog_.at("assets"), basename))
        return true;
    static const std::regex root_weights(
        R"((?:model(?:-\d+(?:-of-\d+)?|_mtp)?)\.safetensors)"
    );
    static const std::regex component_weights(
        R"((?:model|diffusion_pytorch_model)(?:-\d+-of-\d+)?\.safetensors(?:\.index\.json)?)"
    );
    return std::regex_match(
        basename, layout == "components" ? component_weights : root_weights
    );
}

void ModelStore::validate_files(const Json& model, const Json& files) const {
    preparation_check(
        files.is_array() && !files.empty() && files.size() <= 10000,
        "Invalid checkpoint manifest"
    );
    std::set<std::string> names;
    for (const auto& file : files) {
        const auto name = file.at("name").get<std::string>();
        preparation_check(
            allowed(name, model) && names.insert(name).second,
            "Invalid or duplicate checkpoint filename"
        );
        preparation_check(
            file.at("size").is_number_unsigned(), "Invalid checkpoint file size"
        );
        const auto algorithm = file.at("algorithm").get<std::string>();
        preparation_check(
            (algorithm == "sha256" || algorithm == "git-sha1")
                && hex_digest(file.at("digest"), algorithm == "sha256" ? 64 : 40),
            "Invalid checkpoint digest"
        );
    }
    const auto layout = model.at("layout").get<std::string>();
    auto required = model.at("required_files");
    if (layout == "root")
        required = {
            "config.json", "tokenizer_config.json", "model.safetensors.index.json"
        };
    for (const auto& name : required)
        preparation_check(
            names.contains(name.get<std::string>()),
            "Checkpoint is missing required assets"
        );
    if (layout == "composite" || layout == "single_file") {
        preparation_check(
            names.size() == required.size(),
            "Checkpoint manifest differs from pinned assets"
        );
        return;
    }
    const Json paths =
        layout == "components" ? model.at("weight_paths") : Json::array({""});
    for (const auto& prefix : paths) {
        const auto found =
            std::any_of(names.begin(), names.end(), [&](const std::string& name) {
                return fs::path(name).parent_path().generic_string()
                           == prefix.get<std::string>()
                       && name.ends_with(".safetensors");
            });
        preparation_check(found, "Checkpoint has no weights in required component");
    }
}

bool ModelStore::complete(const Json& model) const {
    const auto directory = root_
                           / (model.at("id").get<std::string>() + "-"
                              + model.at("revision").get<std::string>());
    if (!fs::exists(directory / "complete.json"))
        return false;
    try {
        const auto marker = read_document(directory / "complete.json");
        if (marker.at("revision") != model.at("revision"))
            return false;
        validate_files(model, marker.at("files"));
        if (model.at("layout") == "composite"
            && marker.at("files") != model.at("manifest"))
            return false;
        for (const auto& file : marker.at("files")) {
            const auto path = directory / file.at("name").get<std::string>();
            for (auto parent = path; parent != root_ && !parent.empty();
                 parent = parent.parent_path())
                if (fs::is_symlink(parent))
                    return false;
            if (!fs::is_regular_file(path) || fs::file_size(path) != file.at("size"))
                return false;
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

Json ModelStore::manifest(const Json& model, const std::atomic_bool& cancel) const {
    if (model.at("layout") == "composite") {
        validate_files(model, model.at("manifest"));
        return model.at("manifest");
    }
    Transfer metadata;
    if (model.at("layout") == "single_file") {
        const auto size = fetch(fetch_, model.at("asset_url"), metadata, cancel, true);
        preparation_check(size > 0, "Checkpoint size unavailable");
        Json files = Json::array(
            {{{"name", model.at("asset_name")},
              {"size", std::uint64_t(size)},
              {"digest", model.at("revision")},
              {"algorithm", "sha256"}}}
        );
        validate_files(model, files);
        return files;
    }
    const auto repo = model.at("repo_id").get<std::string>(),
               revision = model.at("revision").get<std::string>();
    fetch(
        fetch_,
        "https://huggingface.co/api/models/" + repo + "/revision/" + revision
            + "?blobs=true",
        metadata,
        cancel
    );
    const auto source = Json::parse(metadata.bytes);
    preparation_check(
        source.at("sha").get<std::string>() == revision,
        "Upstream checkpoint revision changed"
    );
    Json files = Json::array();
    for (const auto& item : source.at("siblings")) {
        const auto name = item.at("rfilename").get<std::string>();
        if (!allowed(name, model))
            continue;
        const bool large = item.contains("lfs") && !item.at("lfs").is_null();
        files.push_back(
            {{"name", name},
             {"size", large ? item.at("lfs").at("size") : item.at("size")},
             {"digest", large ? item.at("lfs").at("sha256") : item.at("blobId")},
             {"algorithm", large ? "sha256" : "git-sha1"}}
        );
    }
    validate_files(model, files);
    return files;
}

fs::path ModelStore::acquire(const std::string& model, const std::atomic_bool& cancel) {
    const auto& catalog = entry(model);
    const auto revision = catalog.at("revision").get<std::string>();
    const auto destination = root_ / (model + "-" + revision);
    preparation_cancel(cancel);
    if (complete(catalog))
        return destination;
    fs::create_directories(root_);
    StoreLock lock(root_ / ".download.lock", cancel);
    if (complete(catalog))
        return destination;
    preparation_check(
        !fs::exists(destination),
        "Checkpoint directory is incomplete; inspect model storage.",
        503
    );
    const auto stage = root_ / ("." + model + ".partial");
    fs::remove_all(stage);
    fs::create_directory(stage);
    try {
        const auto files = manifest(catalog, cancel);
        std::uint64_t total = 1024ULL * 1024 * 1024;
        for (const auto& file : files) {
            const auto size = file.at("size").get<std::uint64_t>();
            preparation_check(size <= UINT64_MAX - total, "Checkpoint size overflow");
            total += size;
        }
        preparation_check(
            fs::space(root_).available >= total,
            "Insufficient space for checkpoint",
            503
        );
        total -= 1024ULL * 1024 * 1024;
        std::uint64_t downloaded = 0, reported = 0;
        if (progress_)
            progress_(model, 0, total);
        auto progress = [&](std::size_t bytes) {
            downloaded += bytes;
            if (progress_
                && (downloaded == total || downloaded - reported >= 4 * 1024 * 1024)) {
                progress_(model, downloaded, total);
                reported = downloaded;
            }
        };
        for (const auto& file : files) {
            preparation_cancel(cancel);
            const auto name = file.at("name").get<std::string>();
            const auto url = catalog.at("layout") == "composite"
                                 ? file.at("url").get<std::string>()
                             : catalog.at("layout") == "single_file"
                                 ? catalog.at("asset_url").get<std::string>()
                                 : "https://huggingface.co/"
                                       + catalog.at("repo_id").get<std::string>()
                                       + "/resolve/" + revision + "/" + name;
            download(url, stage / name, file, cancel, fetch_, progress);
        }
        for (const auto& file : files) {
            const auto name = file.at("name").get<std::string>();
            if (!name.ends_with(".safetensors.index.json"))
                continue;
            const auto index = read_document(stage / name);
            preparation_check(
                !index.at("weight_map").empty(), "Empty checkpoint weight index"
            );
            for (const auto& shard : index.at("weight_map")) {
                const auto relative =
                    (fs::path(name).parent_path() / shard.get<std::string>())
                        .generic_string();
                preparation_check(
                    safe_name(relative) && fs::is_regular_file(stage / relative),
                    "Weight index references missing shard"
                );
            }
        }
        preparation_cancel(cancel);
        write_document(
            stage / "complete.json", {{"revision", revision}, {"files", files}}
        );
        fs::rename(stage, destination);
    } catch (...) {
        fs::remove_all(stage);
        throw;
    }
    return destination;
}

std::string ModelStore::selected() const {
    try {
        const auto model =
            read_document(root_ / "selection.json").at("model_id").get<std::string>();
        preparation_check(
            entry(model).at("kind") == "llm", "Invalid selected model", 422
        );
        return model;
    } catch (const std::exception&) {
        throw RequestError("Select a language model before submitting a request.", 422);
    }
}

std::string ModelStore::whisper_model(const std::string& name) const {
    const std::string model = name == "large"   ? "large-v3"
                              : name == "turbo" ? "large-v3-turbo"
                                                : name;
    preparation_check(
        entry("whisper-" + model).at("kind") == "transcription",
        "Unknown Whisper checkpoint",
        422
    );
    return model;
}

std::string ModelStore::whisper_selected() const {
    if (fs::is_regular_file(root_ / "whisper-selection.json")) {
        const auto model = read_document(root_ / "whisper-selection.json")
                               .at("model")
                               .get<std::string>();
        return whisper_model(model);
    }
    // Match the catalogue order used by the Settings selector.
    std::set<std::string> names;
    for (const auto& model : catalog_.at("models"))
        if (model.at("kind") == "transcription")
            names.insert(model.at("id").get<std::string>().substr(8));
    preparation_check(!names.empty(), "No Whisper checkpoint registered", 422);
    return *names.begin();
}

Json ModelStore::context(const std::string& model) const {
    if (!fs::exists(root_ / "context.json"))
        return 65536;
    return read_document(root_ / "context.json").value(model, Json(65536));
}

void ModelStore::select(const std::string& model) {
    preparation_check(
        entry(model).at("kind") == "llm" && complete(entry(model)),
        "Cannot select incomplete language model",
        422
    );
    atomic_document(root_ / "selection.json", {{"model_id", model}});
}

void ModelStore::set_context(const std::string& model, const Json& value) {
    preparation_check(
        entry(model).at("kind") == "llm", "This checkpoint is not a language model", 422
    );
    preparation_check(
        value.is_null()
            || (value.is_number_integer() && value > 0 && value <= 2147483647),
        "Invalid context limit",
        422
    );
    const auto maximum = architecture_context(model);
    preparation_check(
        value.is_null() || maximum.is_null() || value <= maximum,
        "Context limit exceeds the checkpoint architecture maximum",
        422
    );
    fs::create_directories(root_);
    auto values = fs::exists(root_ / "context.json")
                      ? read_document(root_ / "context.json")
                      : Json::object();
    values[model] = value;
    atomic_document(root_ / "context.json", values);
}

void ModelStore::select_whisper(const std::string& name) {
    const auto model = whisper_model(name);
    fs::create_directories(root_);
    atomic_document(root_ / "whisper-selection.json", {{"model", model}});
}

Json ModelStore::architecture_context(const std::string& model) const {
    const auto& item = entry(model);
    if (item.at("kind") != "llm" || !complete(item))
        return nullptr;
    const auto config = read_document(
        root_ / (model + "-" + item.at("revision").get<std::string>()) / "config.json"
    );
    const auto text = config.value("text_config", config);
    const auto maximum = text.value("max_position_embeddings", Json());
    return maximum.is_number_integer() && maximum > 0 ? maximum : Json(nullptr);
}

Json ModelStore::status() const {
    Json models = Json::array();
    for (const auto& item : catalog_.at("models")) {
        const auto id = item.at("id").get<std::string>();
        Json model;
        for (const auto* field :
             {"id",
              "repo_id",
              "revision",
              "license",
              "estimated_bytes",
              "kind",
              "display_name",
              "license_url",
              "license_notice",
              "inference_available"})
            model[field] = item.value(field, Json());
        model["status"] = complete(item) ? "complete" : "not_downloaded";
        model["downloaded_bytes"] = 0;
        model["total_bytes"] = 0;
        model["error"] = nullptr;
        model["context_limit"] = nullptr;
        model["architecture_context_limit"] = nullptr;
        try {
            model["context_limit"] =
                item.at("kind") == "llm" ? context(id) : Json(nullptr);
            model["architecture_context_limit"] = architecture_context(id);
        } catch (const std::exception& error) {
            model["status"] = "failed";
            model["error"] = error.what();
        }
        models.push_back(std::move(model));
    }
    Json selection = nullptr, whisper = nullptr;
    try {
        selection = selected();
    } catch (const RequestError&) {
    }
    try {
        whisper = whisper_selected();
    } catch (const RequestError&) {
    }
    return {
        {"models", models},
        {"selected_model_id", selection},
        {"selected_whisper", whisper}
    };
}
}  // namespace kadan::serving
