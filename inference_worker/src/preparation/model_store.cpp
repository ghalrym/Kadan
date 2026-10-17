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

#include "../weights/metadata_json.hpp"
#include "kadan/gguf_text.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
namespace {
void validate_text_checkpoint(const fs::path& checkpoint) {
    try {
        auto budget = std::make_shared<checkpoint::MemoryBudget>(4ULL * 1024 * 1024);
        text::validate_config(checkpoint.string().c_str(), budget);
    } catch (const std::exception& error) {
        throw RequestError(
            std::string("Unsupported text checkpoint: ") + error.what(), 422
        );
    }
}

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
    bool head = false,
    std::uint64_t offset = 0
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
    preparation_check(offset <= INT64_MAX, "Checkpoint resume offset exceeds limit");
    curl_easy_setopt(curl.get(), CURLOPT_RESUME_FROM_LARGE, curl_off_t(offset));
    const auto status = curl_easy_perform(curl.get());
    preparation_cancel(cancel);
    preparation_check(
        status != CURLE_RANGE_ERROR,
        "Checkpoint server cannot resume; partial retained",
        503
    );
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
    bool head = false,
    std::uint64_t offset = 0
) {
    if (!source)
        return transfer(url, target, cancel, head, offset);
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
        head,
        offset
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
    const ManifestFile& item,
    const std::atomic_bool& cancel,
    const ModelStore::Fetch& source,
    const std::function<void(std::size_t)>& progress
) {
    const auto size = item.size;
    const bool sha256 = item.algorithm == "sha256";
    preparation_check(
        sha256 || item.algorithm == "git-sha1", "Unknown digest algorithm"
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
    const auto offset = fs::exists(path) ? fs::file_size(path) : 0;
    preparation_check(offset <= size, "Checkpoint partial exceeds expected size");
    if (offset) {
        std::ifstream input(path, std::ios::binary);
        std::array<char, 1024 * 1024> buffer{};
        std::uint64_t remaining = offset;
        while (remaining) {
            preparation_cancel(cancel);
            const auto count = std::min<std::uint64_t>(remaining, buffer.size());
            input.read(buffer.data(), count);
            preparation_check(bool(input), "Cannot read checkpoint partial", 503);
            preparation_check(
                EVP_DigestUpdate(digest.get(), buffer.data(), count) == 1,
                "Digest update failed"
            );
            remaining -= count;
        }
    }
    std::ofstream output(path, std::ios::binary | std::ios::app);
    preparation_check(bool(output), "Cannot write checkpoint", 503);
    Transfer target;
    target.output = &output;
    target.progress = progress;
    target.digest = digest.get();
    target.maximum = size;
    target.received = offset;
    // Report saved bytes only after reading them into the verification digest.
    progress(offset);
    if (offset < size)
        fetch(source, url, target, cancel, false, offset);
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
    if (actual != item.digest) {
        // Only the corrupt staged artifact is discarded; verified siblings survive.
        fs::remove(path);
        throw RequestError(
            "Checkpoint integrity verification failed: " + path.filename().string()
        );
    }
}

void atomic_document(const fs::path& destination, const Json& value) {
    const auto temporary = destination.string() + ".worker.tmp";
    write_document(temporary, value);
    fs::rename(temporary, destination);
}
}  // namespace

void ManifestFile::validate() const {
    preparation_check(safe_name(name), "Invalid checkpoint filename");
    preparation_check(
        (algorithm == "sha256" || algorithm == "git-sha1")
            && hex_digest(digest, algorithm == "sha256" ? 64 : 40),
        "Invalid checkpoint digest"
    );
}

ManifestFile ManifestFile::parse(const Json& value) {
    preparation_check(
        value.at("size").is_number_unsigned(), "Invalid checkpoint file size"
    );
    ManifestFile file{
        value.at("name").get<std::string>(),
        value.at("size").get<std::uint64_t>(),
        value.at("digest").get<std::string>(),
        value.at("algorithm").get<std::string>(),
        value.value("url", std::string())
    };
    file.validate();
    return file;
}

void to_json(Json& value, const ManifestFile& file) {
    value = {
        {"name", file.name},
        {"size", file.size},
        {"digest", file.digest},
        {"algorithm", file.algorithm}
    };
    if (!file.url.empty())
        value["url"] = file.url;
}

std::vector<ManifestFile> parse_manifest(const Json& value) {
    preparation_check(
        value.is_array() && value.size() <= 10000, "Invalid checkpoint manifest"
    );
    std::vector<ManifestFile> files;
    for (const auto& item : value)
        files.push_back(ManifestFile::parse(item));
    return files;
}

CatalogEntry CatalogEntry::parse(const Json& value) {
    CatalogEntry entry;
    entry.id = value.at("id");
    entry.repo_id = value.at("repo_id");
    entry.revision = value.at("revision");
    entry.kind = value.at("kind");
    entry.layout = value.at("layout");
    preparation_check(
        entry.layout == "root" || entry.layout == "components"
            || entry.layout == "composite" || entry.layout == "single_file"
            || entry.layout == "gguf",
        "Invalid checkpoint layout"
    );
    preparation_check(
        safe_name(entry.id) && !fs::path(entry.id).has_parent_path()
            && safe_name(entry.revision) && !fs::path(entry.revision).has_parent_path(),
        "Invalid checkpoint identity"
    );
    entry.license = value.value("license", std::string());
    entry.estimated_bytes = value.value("estimated_bytes", std::uint64_t(0));
    const auto optional_text = [&](const char* name) -> std::optional<std::string> {
        const auto found = value.find(name);
        if (found == value.end() || found->is_null())
            return std::nullopt;
        return found->get<std::string>();
    };
    entry.display_name = optional_text("display_name");
    entry.license_url = optional_text("license_url");
    entry.license_notice = optional_text("license_notice");
    entry.asset_url = optional_text("asset_url").value_or("");
    entry.asset_name = optional_text("asset_name").value_or("");
    entry.inference_available = value.at("inference_available");
    entry.required_files = value.at("required_files").get<std::vector<std::string>>();
    entry.component_paths = value.value("component_paths", std::vector<std::string>());
    entry.weight_paths = value.value("weight_paths", std::vector<std::string>());
    entry.manifest = parse_manifest(value.at("manifest"));
    // The first manifest file carries the GGUF metadata: one unsplit file, or a
    // tensor-free first split acquired before any weight split.
    if (entry.layout == "gguf")
        preparation_check(
            entry.kind == "llm" && !entry.manifest.empty()
                && !fs::path(entry.asset_name).has_parent_path()
                && entry.manifest.front().name == entry.asset_name,
            "Invalid GGUF catalog metadata"
        );
    return entry;
}

HardwareTier HardwareTier::parse(const Json& value) {
    HardwareTier tier{
        value.at("id"), value.at("name"), value.at("vram_gb"), value.at("ram_gb"), {}
    };
    preparation_check(
        value.at("models").is_object() && value.at("models").size() == 6,
        "Invalid hardware tier models"
    );
    for (const auto& [kind, model] : value.at("models").items())
        tier.models.emplace(
            kind,
            model.is_null() ? std::nullopt : std::optional(model.get<std::string>())
        );
    return tier;
}

ModelStore::ModelStore(
    fs::path root, fs::path data, Fetch fetch, DownloadProgress progress
)
    : root_(std::move(root)), data_(std::move(data)), fetch_(std::move(fetch)),
      progress_(std::move(progress)) {
    const auto document = read_document(data_ / "catalog.json");
    assets_ = document.at("assets").get<std::vector<std::string>>();
    std::set<std::string> identities;
    for (const auto& value : document.at("models")) {
        auto model = CatalogEntry::parse(value);
        preparation_check(
            identities.insert(model.id).second, "Duplicate catalog model"
        );
        catalog_.push_back(std::move(model));
    }
    std::set<std::string> tier_ids;
    for (const auto& value : document.value("tiers", Json::array())) {
        auto tier = HardwareTier::parse(value);
        preparation_check(tier_ids.insert(tier.id).second, "Duplicate hardware tier");
        for (const auto& [kind, model] : tier.models)
            preparation_check(
                !model || entry(*model).kind == kind,
                "Hardware tier model kind mismatch"
            );
        preparation_check(
            tier.models.at("llm").has_value(), "Hardware tier has no LLM"
        );
        tiers_.push_back(std::move(tier));
    }
}

const CatalogEntry& ModelStore::entry(const std::string& model) const {
    for (const auto& candidate : catalog_)
        if (candidate.id == model)
            return candidate;
    throw RequestError("Unknown catalog model: " + model, 422);
}

bool ModelStore::allowed(const std::string& name, const CatalogEntry& model) const {
    if (!safe_name(name))
        return false;
    const auto layout = model.layout;
    const auto& required = model.required_files;
    const auto contains = [](const std::vector<std::string>& values,
                             const std::string& value) {
        return std::find(values.begin(), values.end(), value) != values.end();
    };
    if (layout == "composite" || layout == "gguf")
        return contains(required, name);
    if (layout == "single_file")
        return model.asset_name == name && fs::path(name).parent_path().empty();
    auto basename = name;
    if (layout == "components") {
        if (contains(required, name) || name == "LICENSE" || name == "LICENSE.txt"
            || name == "NOTICE" || name == "README.md")
            return true;
        auto parent = fs::path(name).parent_path().generic_string();
        if (parent.empty())
            parent = ".";
        if (!contains(model.component_paths, parent))
            return false;
        basename = fs::path(name).filename().string();
    } else if (fs::path(name).has_parent_path())
        return false;
    if (contains(assets_, basename))
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

void ModelStore::validate_files(
    const CatalogEntry& model, const std::vector<ManifestFile>& files
) const {
    preparation_check(
        !files.empty() && files.size() <= 10000, "Invalid checkpoint manifest"
    );
    std::set<std::string> names;
    for (const auto& file : files) {
        const auto name = file.name;
        preparation_check(
            allowed(name, model) && name != "resume.json" && name != "complete.json"
                && names.insert(name).second,
            "Invalid or duplicate checkpoint filename"
        );
        file.validate();
    }
    const auto layout = model.layout;
    auto required = model.required_files;
    if (layout == "root")
        required = {
            "config.json", "tokenizer_config.json", "model.safetensors.index.json"
        };
    for (const auto& name : required)
        preparation_check(
            names.contains(name), "Checkpoint is missing required assets"
        );
    if (layout == "composite" || layout == "single_file" || layout == "gguf") {
        preparation_check(
            names.size() == required.size(),
            "Checkpoint manifest differs from pinned assets"
        );
        if (layout == "gguf") {
            preparation_check(
                names.contains(model.asset_name), "Invalid GGUF split manifest"
            );
            for (const auto& file : files)
                preparation_check(
                    file.name.ends_with(".gguf")
                        && !fs::path(file.name).has_parent_path()
                        && file.url.starts_with("https://")
                        && (files.size() == 1 || file.name != model.asset_name
                            || file.size <= 32 * 1024 * 1024),
                    "Invalid GGUF split asset"
                );
        }
        return;
    }
    const auto paths =
        layout == "components" ? model.weight_paths : std::vector<std::string>{""};
    for (const auto& prefix : paths) {
        const auto found =
            std::any_of(names.begin(), names.end(), [&](const std::string& name) {
                return fs::path(name).parent_path().generic_string() == prefix
                       && name.ends_with(".safetensors");
            });
        preparation_check(found, "Checkpoint has no weights in required component");
    }
}

bool ModelStore::complete(const CatalogEntry& model) const {
    const auto directory = root_ / (model.id + "-" + model.revision);
    if (!fs::exists(directory / "complete.json"))
        return false;
    try {
        const auto marker = read_document(directory / "complete.json");
        if (marker.at("revision") != model.revision)
            return false;
        const auto files = parse_manifest(marker.at("files"));
        validate_files(model, files);
        if ((model.layout == "composite" || model.layout == "gguf")
            && files != model.manifest)
            return false;
        if (model.layout == "single_file"
            && (files.size() != 1 || files.front().algorithm != "sha256"
                || files.front().digest != model.revision))
            return false;
        for (const auto& file : files) {
            const auto path = directory / file.name;
            for (auto parent = path; parent != root_ && !parent.empty();
                 parent = parent.parent_path())
                if (fs::is_symlink(parent))
                    return false;
            if (!fs::is_regular_file(path) || fs::file_size(path) != file.size)
                return false;
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<ManifestFile>
ModelStore::manifest(const CatalogEntry& model, const std::atomic_bool& cancel) const {
    if (model.layout == "composite" || model.layout == "gguf") {
        validate_files(model, model.manifest);
        return model.manifest;
    }
    Transfer metadata;
    if (model.layout == "single_file") {
        const auto size = fetch(fetch_, model.asset_url, metadata, cancel, true);
        preparation_check(size > 0, "Checkpoint size unavailable");
        std::vector<ManifestFile> files = {
            {model.asset_name, std::uint64_t(size), model.revision, "sha256", ""}
        };
        validate_files(model, files);
        return files;
    }
    const auto repo = model.repo_id, revision = model.revision;
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
    std::vector<ManifestFile> files;
    for (const auto& item : source.at("siblings")) {
        const auto name = item.at("rfilename").get<std::string>();
        if (!allowed(name, model))
            continue;
        const bool large = item.contains("lfs") && !item.at("lfs").is_null();
        files.push_back(
            ManifestFile::parse(
                {{"name", name},
                 {"size", large ? item.at("lfs").at("size") : item.at("size")},
                 {"digest", large ? item.at("lfs").at("sha256") : item.at("blobId")},
                 {"algorithm", large ? "sha256" : "git-sha1"}}
            )
        );
    }
    validate_files(model, files);
    return files;
}

fs::path ModelStore::acquire(const std::string& model, const std::atomic_bool& cancel) {
    const auto& catalog = entry(model);
    const auto revision = catalog.revision;
    const auto destination = root_ / (model + "-" + revision);
    preparation_cancel(cancel);
    if (complete(catalog)) {
        if (catalog.layout == "gguf")
            prepare_gguf_assets(destination, catalog.asset_name, cancel);
        if (catalog.kind == "llm")
            validate_text_checkpoint(destination);
        return destination;
    }
    fs::create_directories(root_);
    StoreLock lock(root_ / ".download.lock", cancel);
    if (complete(catalog)) {
        if (catalog.layout == "gguf")
            prepare_gguf_assets(destination, catalog.asset_name, cancel);
        if (catalog.kind == "llm")
            validate_text_checkpoint(destination);
        return destination;
    }
    preparation_check(
        !fs::exists(destination),
        "Checkpoint directory is incomplete; inspect model storage.",
        503
    );
    const auto stage = root_ / ("." + model + ".partial");
    preparation_check(
        !fs::is_symlink(stage), "Checkpoint staging directory is a symlink"
    );
    preparation_check(
        !fs::is_symlink(stage / "complete.json"),
        "Checkpoint completion marker is a symlink"
    );
    const auto files = manifest(catalog, cancel);
    auto ordered = files;
    // Validate all index metadata before transferring any tensor payloads.
    std::stable_partition(ordered.begin(), ordered.end(), [](const auto& file) {
        return file.name.ends_with(".safetensors.index.json");
    });
    if (catalog.kind == "llm")
        std::stable_partition(ordered.begin(), ordered.end(), [](const auto& file) {
            return file.name == "config.json";
        });
    if (catalog.layout == "gguf")
        std::stable_partition(ordered.begin(), ordered.end(), [&](const auto& file) {
            return file.name == catalog.asset_name;
        });
    std::set<std::string> names;
    for (const auto& file : files) {
        names.insert(file.name);
        preparation_check(
            catalog.kind != "llm" || file.name != "config.json"
                || file.size <= 16 * 1024 * 1024,
            "Text checkpoint config exceeds size limit",
            422
        );
        preparation_check(
            !file.name.ends_with(".safetensors.index.json")
                || file.size <= 32 * 1024 * 1024,
            "Checkpoint weight index exceeds size limit"
        );
    }
    preparation_check(
        catalog.kind != "llm" || names.contains("config.json")
            || (catalog.layout == "gguf" && names.contains(catalog.asset_name)),
        "Text checkpoint is missing config.json",
        422
    );
    auto identity_files = files;
    std::sort(
        identity_files.begin(),
        identity_files.end(),
        [](const auto& left, const auto& right) { return left.name < right.name; }
    );
    const Json identity = {
        {"id", catalog.id},
        {"repo_id", catalog.repo_id},
        {"revision", revision},
        {"layout", catalog.layout},
        {"asset_url", catalog.asset_url},
        {"files", identity_files}
    };
    if (fs::exists(stage / "resume.json")) {
        preparation_check(
            read_document(stage / "resume.json") == identity,
            "Checkpoint partial identity changed; inspect model storage.",
            503
        );
    } else {
        fs::create_directory(stage);
        // A process can stop between creating the stage and publishing its identity.
        // Only an otherwise empty stage is safe to initialize without an identity.
        for (const auto& item : fs::directory_iterator(stage))
            preparation_check(
                item.path().filename() == "resume.json.worker.tmp" && !item.is_symlink()
                    && item.is_regular_file(),
                "Checkpoint partial has no identity; inspect model storage.",
                503
            );
        atomic_document(stage / "resume.json", identity);
    }
    std::uint64_t total = 1024ULL * 1024 * 1024, saved = 0;
    for (const auto& file : files) {
        const auto path = stage / file.name;
        for (auto parent = path; parent != root_ && !parent.empty();
             parent = parent.parent_path())
            preparation_check(
                !fs::is_symlink(parent), "Checkpoint partial is a symlink"
            );
        if (fs::exists(path)) {
            preparation_check(
                fs::is_regular_file(path), "Checkpoint partial is not a file"
            );
            const auto size = fs::file_size(path);
            preparation_check(
                size <= file.size, "Checkpoint partial exceeds expected size"
            );
            preparation_check(size <= UINT64_MAX - saved, "Checkpoint size overflow");
            saved += size;
        }
        const auto size = file.size;
        preparation_check(size <= UINT64_MAX - total, "Checkpoint size overflow");
        total += size;
    }
    preparation_check(
        fs::space(root_).available >= total - saved,
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
    for (const auto& file : ordered) {
        preparation_cancel(cancel);
        const auto name = file.name;
        const auto url = (catalog.layout == "composite" || catalog.layout == "gguf")
                             ? file.url
                         : catalog.layout == "single_file"
                             ? catalog.asset_url
                             : "https://huggingface.co/" + catalog.repo_id + "/resolve/"
                                   + revision + "/" + name;
        download(url, stage / name, file, cancel, fetch_, progress);
        if (catalog.layout == "gguf" && name == catalog.asset_name)
            prepare_gguf_assets(stage, name, cancel);
        if (catalog.kind == "llm" && name == "config.json")
            validate_text_checkpoint(stage);
        if (!name.ends_with(".safetensors.index.json"))
            continue;
        const auto index = read_document(stage / name);
        preparation_check(
            index.at("weight_map").is_object() && !index.at("weight_map").empty(),
            "Invalid checkpoint weight index"
        );
        for (const auto& shard : index.at("weight_map")) {
            const auto relative =
                (fs::path(name).parent_path() / shard.get<std::string>())
                    .generic_string();
            preparation_check(
                safe_name(relative) && relative.ends_with(".safetensors")
                    && names.contains(relative),
                "Weight index references missing or unsupported shard"
            );
        }
    }
    preparation_cancel(cancel);
    write_document(stage / "complete.json", {{"revision", revision}, {"files", files}});
    fs::rename(stage, destination);
    return destination;
}

std::optional<std::string> ModelStore::tier() const {
    if (!fs::is_regular_file(root_ / "tier.json"))
        return std::nullopt;
    auto id = read_document(root_ / "tier.json").at("tier").get<std::string>();
    hardware_tier(id);
    return id;
}

const HardwareTier& ModelStore::hardware_tier(const std::string& id) const {
    for (const auto& tier : tiers_)
        if (tier.id == id)
            return tier;
    throw RequestError("Unknown hardware tier: " + id, 422);
}

void ModelStore::select_tier(const std::string& id) {
    hardware_tier(id);
    fs::create_directories(root_);
    atomic_document(root_ / "tier.json", {{"tier", id}});
}

std::string ModelStore::selected() const {
    std::optional<std::string> id;
    try {
        id = tier();
    } catch (const std::exception&) {
    }
    if (!id)
        throw RequestError(
            "Choose a hardware tier in Settings before submitting a request.", 422
        );
    return *hardware_tier(*id).models.at("llm");
}

std::string ModelStore::whisper_model(const std::string& name) const {
    const std::string model = name == "large"   ? "large-v3"
                              : name == "turbo" ? "large-v3-turbo"
                                                : name;
    preparation_check(
        entry("whisper-" + model).kind == "transcription",
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
    // Every hardware tier downloads Whisper medium for transcription.
    return whisper_model("medium");
}

Json ModelStore::context(const std::string& model) const {
    if (!fs::exists(root_ / "context.json"))
        return 65536;
    return read_document(root_ / "context.json").value(model, Json(65536));
}

void ModelStore::set_context(const std::string& model, const Json& value) {
    preparation_check(
        entry(model).kind == "llm", "This checkpoint is not a language model", 422
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

bool ModelStore::whisper_formatting() const {
    const auto path = root_ / "whisper-formatting.json";
    return !fs::exists(path) || read_document(path).at("enabled").get<bool>();
}

void ModelStore::set_whisper_formatting(bool enabled) {
    fs::create_directories(root_);
    atomic_document(root_ / "whisper-formatting.json", {{"enabled", enabled}});
}

Json ModelStore::architecture_context(const std::string& model) const {
    const auto& item = entry(model);
    if (item.kind != "llm" || !complete(item))
        return nullptr;
    const auto config =
        read_document(root_ / (model + "-" + item.revision) / "config.json");
    const auto text = config.value("text_config", config);
    const auto maximum = text.value("max_position_embeddings", Json());
    return maximum.is_number_integer() && maximum > 0 ? maximum : Json(nullptr);
}

Json ModelStore::status() const {
    Json models = Json::array();
    for (const auto& item : catalog_) {
        const auto id = item.id;
        const auto text = [](const std::optional<std::string>& value) -> Json {
            return value ? Json(*value) : Json(nullptr);
        };
        Json model = {
            {"id", item.id},
            {"repo_id", item.repo_id},
            {"revision", item.revision},
            {"license", item.license},
            {"estimated_bytes", item.estimated_bytes},
            {"kind", item.kind},
            {"display_name", text(item.display_name)},
            {"license_url", text(item.license_url)},
            {"license_notice", text(item.license_notice)},
            {"inference_available", item.inference_available}
        };
        model["status"] = complete(item) ? "complete" : "not_downloaded";
        model["downloaded_bytes"] = 0;
        model["total_bytes"] = 0;
        model["error"] = nullptr;
        model["context_limit"] = nullptr;
        model["architecture_context_limit"] = nullptr;
        try {
            model["context_limit"] = item.kind == "llm" ? context(id) : Json(nullptr);
            model["architecture_context_limit"] = architecture_context(id);
        } catch (const std::exception& error) {
            model["status"] = "failed";
            model["error"] = error.what();
        }
        models.push_back(std::move(model));
    }
    Json selection = nullptr, whisper = nullptr, tier = nullptr;
    try {
        selection = selected();
        tier = *this->tier();
    } catch (const RequestError&) {
    }
    try {
        whisper = whisper_selected();
    } catch (const RequestError&) {
    }
    Json tiers = Json::array();
    for (const auto& item : tiers_) {
        Json assigned = Json::object();
        for (const auto& [kind, model] : item.models)
            assigned[kind] = model ? Json(*model) : Json(nullptr);
        tiers.push_back(
            {{"id", item.id},
             {"name", item.name},
             {"vram_gb", item.vram_gb},
             {"ram_gb", item.ram_gb},
             {"models", assigned}}
        );
    }
    return {
        {"models", models},
        {"tiers", tiers},
        {"selected_tier", tier},
        {"selected_model_id", selection},
        {"selected_whisper", whisper},
        {"whisper_formatting", whisper_formatting()}
    };
}
}  // namespace kadan::serving
