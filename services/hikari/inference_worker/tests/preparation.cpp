#include "kadan/preparation.hpp"

#include <algorithm>
#include <archive.h>
#include <archive_entry.h>
#include <array>
#include <barrier>
#include <bit>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>


using namespace kadan::serving;

namespace {
const std::atomic_bool running{false};

struct Directory {
    fs::path path;

    Directory() {
        std::string pattern = "/tmp/kadan-preparation-unit-XXXXXX";
        auto* value = mkdtemp(pattern.data());
        assert(value);
        path = value;
    }

    ~Directory() {
        fs::remove_all(path);
    }
};

void bytes(const fs::path& path, std::string_view value) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(value.data(), value.size());
    assert(out.good());
}

void rejects(const std::function<void()>& action) {
    bool failed = false;
    try {
        action();
    } catch (const std::exception&) {
        failed = true;
    }
    assert(failed);
}

Json entry(const std::string& id, const Json& manifest) {
    Json required = Json::array();
    for (const auto& item : manifest)
        required.push_back(item.at("name"));
    return {
        {"id", id},
        {"revision", "revision"},
        {"repo_id", "test/model"},
        {"kind", "image"},
        {"layout", "composite"},
        {"inference_available", true},
        {"required_files", required},
        {"manifest", manifest}
    };
}

Json local_manifest(
    const fs::path& directory, const std::map<std::string, std::string>& content
) {
    Json manifest = Json::array();
    for (const auto& [name, value] : content) {
        bytes(directory / name, value);
        manifest.push_back(
            {{"name", name},
             {"size", value.size()},
             {"digest", file_digest(directory / name, running)},
             {"algorithm", "sha256"},
             {"url", "https://unit.invalid/" + name}}
        );
    }
    return manifest;
}

void catalog(const fs::path& data, const Json& models) {
    fs::create_directories(data);
    write_document(
        data / "catalog.json", {{"models", models}, {"assets", Json::array()}}
    );
}

InferenceRequest request(
    const std::string& id, const Json& body, const std::string& operation = "generate"
) {
    auto result = parse_request(
        {{"id", std::string(32, 'a')},
         {"command", "submit"},
         {"operation", operation},
         {"input", body}}
    );
    result.id = id;
    return result;
}

void pinned_catalog_units() {
    Directory tmp;
    ModelStore store(tmp.path, worker_data_directory());
    assert(store.whisper_selected() == "medium");
    store.select_whisper("medium");
    assert(store.whisper_selected() == "medium");
    for (const auto* name : {"unknown", "large", "turbo"})
        rejects([&] { store.whisper_model(name); });
    assert(
        worker_requirement(kadan::Workload::speech, std::nullopt, store)["model"]
        == "whisper-medium"
    );
    const auto speech = worker_requirement(kadan::Workload::speech, "medium", store);
    assert(speech["minimum"][1] == speech["device_floor"]);
    assert(
        worker_requirement(
            kadan::Workload::decision, std::nullopt, store
        )["device_floor"]
        == 0
    );
    assert(
        worker_requirement(kadan::Workload::image, std::nullopt, store)["model"]
        == "qwen-image-2.1-turbo"
    );
    assert(
        worker_requirement(kadan::Workload::video, std::nullopt, store)["model"]
        == "h3-fl2va-int8-turbo"
    );
    for (const auto workload :
         {kadan::Workload::llm, kadan::Workload::tts, kadan::Workload::formatting})
        rejects([&] { worker_requirement(workload, std::nullopt, store); });
    for (const auto* model :
         {"qwen-image-2.1-turbo", "h3-fl2va-int8-turbo", "whisper-medium", "laya"})
        for (const auto* path :
             {"../config.json", "/config.json", "model.py", "processor/run.py"})
            assert(!store.allowed(path, store.entry(model)));
    const auto& video = store.entry("h3-fl2va-int8-turbo");
    store.validate_files(video, video.manifest);
    assert(!store.allowed("FL2VA/transformer/model-00001-of-00013.safetensors", video));
}

void shared_settings_units() {
    Directory tmp;
    ModelStore store(tmp.path, worker_data_directory());
    constexpr std::size_t workers = 3;
    int start[2];
    assert(pipe(start) == 0);
    std::vector<pid_t> children;
    for (std::size_t index = 0; index < workers; ++index) {
        const auto child = fork();
        assert(child >= 0);
        if (child == 0) {
            close(start[1]);
            char signal = 0;
            if (read(start[0], &signal, 1) != 1)
                _exit(1);
            close(start[0]);
            try {
                ModelStore worker(tmp.path, worker_data_directory());
                for (int repeat = 0; repeat < 10; ++repeat) {
                    worker.select_whisper("medium");
                    worker.set_whisper_formatting(false);
                }
                _exit(0);
            } catch (const std::exception&) {
                _exit(1);
            }
        }
        children.push_back(child);
    }
    close(start[0]);
    assert(write(start[1], "xxx", workers) == ssize_t(workers));
    close(start[1]);
    bool succeeded = true;
    for (const auto child : children) {
        int status = 0;
        const auto waited = waitpid(child, &status, 0);
        succeeded &= waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    assert(succeeded);
    assert(store.whisper_selected() == "medium");
    assert(!store.whisper_formatting());
}

void whisper_formatting_setting_units() {
    Directory tmp;
    ModelStore store(tmp.path, worker_data_directory());
    store.select_whisper("medium");
    assert(store.whisper_formatting());
    for (const bool enabled : {false, true}) {
        auto configuration = request(
            "formatting",
            {{"feature", "whisper_formatting"}, {"enabled", enabled}},
            "configure"
        );
        prepare_input(
            std::get<WhisperFormattingRequest>(configuration.input),
            configuration,
            store,
            running
        );
        ModelStore reopened(tmp.path, worker_data_directory());
        assert(reopened.whisper_formatting() == enabled);
        assert(reopened.status().at("whisper_formatting") == enabled);
        assert(reopened.whisper_selected() == "medium");
    }
    rejects([&] {
        request("invalid", {{"feature", "whisper_formatting"}, {"enabled", true}});
    });
}

void whisper_formatting_request_units() {
    Directory tmp;
    ModelStore store(tmp.path, worker_data_directory());
    store.select_whisper("medium");
    // Preparation checks file presence only; no model is loaded or executed.
    const auto root = tmp.path / "whisper/medium";
    for (const auto* file :
         {"model.safetensors", "dimensions.txt", "assets/english.tokens"})
        bytes(root / file, "");
    for (const bool saved : {false, true}) {
        store.set_whisper_formatting(saved);
        for (const Json& override : {Json(), Json(false), Json(true)}) {
            auto transcription = request(
                "transcription",
                {{"feature", "stt"},
                 {"audio",
                  "data:audio/"
                  "wav;base64,"
                  "UklGRiYAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQIAAAAAAA=="},
                 {"formatting", override}}
            );
            auto& input = std::get<TranscriptionRequest>(transcription.input);
            assert(input.formatting.has_value() == !override.is_null());
            prepare_input(input, transcription, store, running);
            const bool expected = override.is_null() ? saved : override.get<bool>();
            assert(input.formatting == expected);
            assert(transcription.model == "medium");
            const auto result = present_transcription(transcription, "  Hello.  ");
            assert(result.at("text") == "  Hello.  ");
            assert(result.at("raw_text") == "  Hello.  ");
            assert(
                result.at("formatting_status") == (expected ? "requested" : "disabled")
            );
        }
        auto omitted = request(
            "omitted",
            {{"feature", "stt"},
             {"audio",
              "data:audio/"
              "wav;base64,"
              "UklGRiYAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQIAAAAAAA=="}}
        );
        auto& input = std::get<TranscriptionRequest>(omitted.input);
        prepare_input(input, omitted, store, running);
        assert(input.formatting == saved);
    }
}

void whisper_formatting_response_units() {
    auto transcription = request(
        "format", {{"feature", "stt"}, {"audio", "unused"}, {"formatting", true}}
    );
    transcription.model = "medium";
    const std::string raw = "um send it on friday no thursday";
    const auto result = present_transcription(transcription, raw);
    assert(result.at("text") == raw && result.at("raw_text") == raw);
    assert(result.at("formatting_status") == "requested");
    assert(result.at("formatting_model").is_null());
}

void checkpoint_catalog_units() {
    Directory tmp;
    unsigned transfers = 0;
    ModelStore store(
        tmp.path,
        worker_data_directory(),
        [&](const auto&, const auto&, const auto&, bool, std::uint64_t)
            -> std::int64_t {
            ++transfers;
            throw RequestError("Unexpected checkpoint transfer");
        }
    );
    const auto& turbo = store.entry("qwen-image-2.1-turbo");
    assert(turbo.kind == "image" && turbo.inference_available);
    assert(turbo.repo_id == "Qwen/Qwen-Image-2.1-Turbo");
    assert(turbo.revision == "fc13ba1986fcb905854dc8de2f52eee395f49b5b");
    assert(turbo.license == "qwen-research" && turbo.license_notice);
    assert(
        turbo.license_notice->find("research and evaluation only") != std::string::npos
    );
    assert(
        turbo.license_url
        == "https://huggingface.co/" + turbo.repo_id + "/blob/" + turbo.revision
               + "/LICENSE"
    );
    assert(store.allowed("text_encoder/model.safetensors", turbo));
    assert(store.allowed("processor/processor_config.json", turbo));
    assert(!store.allowed("processor/run.py", turbo));
    assert(
        std::find(
            turbo.required_files.begin(),
            turbo.required_files.end(),
            "text_encoder/model.safetensors.index.json"
        )
        == turbo.required_files.end()
    );
    for (const auto* removed :
         {"qwen3.6-35b-a3b-q4-k-m",
          "qwen3.8-flash-next-ud-iq3-xxs",
          "qwen3.8-flash-next-q3-k-xl",
          "qwen-tts-1.7b-custom",
          "superwhisper-s1-mini",
          "h3-fl2va",
          "qwen-image-2.1"})
        rejects([&] { store.entry(removed); });
    const auto status = store.status();
    assert(status.at("models").size() == 4);
    assert(!status.contains("tiers") && !status.contains("selected_tier"));
    assert(status.at("selected_model_id") == "lfm3-1.7b-bf16");
    assert(transfers == 0);
}

void manifest_model_units() {
    const Json source = {
        {"name", "config.json"},
        {"size", std::uint64_t(2)},
        {"digest", std::string(64, 'a')},
        {"algorithm", "sha256"},
        {"url", "https://unit.invalid/config.json"}
    };
    const auto file = ManifestFile::parse(source);
    assert(Json(file) == source);
    for (const auto& replacement : {Json(-1), Json(1.5), Json("2")}) {
        auto invalid = source;
        invalid["size"] = replacement;
        rejects([&] { ManifestFile::parse(invalid); });
    }
    auto invalid = source;
    invalid["digest"] = "invalid";
    rejects([&] { ManifestFile::parse(invalid); });
    invalid = source;
    invalid["name"] = "../config.json";
    rejects([&] { ManifestFile::parse(invalid); });
    auto model = entry("test", Json::array({source}));
    model["layout"] = "unknown";
    rejects([&] { CatalogEntry::parse(model); });
    model["layout"] = "gguf";
    rejects([&] { CatalogEntry::parse(model); });
    model["layout"] = "composite";
    for (const auto* kind : {"llm", "speech", "formatting"}) {
        model["kind"] = kind;
        rejects([&] { CatalogEntry::parse(model); });
    }
    model["kind"] = "image";
    model["id"] = "../test";
    rejects([&] { CatalogEntry::parse(model); });
}

void typed_request_units() {
    const auto decision = request(
        "decision",
        {{"feature", "decisions"},
         {"state", "state"},
         {"questions",
          Json::array(
              {{{"key", "yes"},
                {"type", "Noul"},
                {"instructions", "Decide"},
                {"true_when", "ready"},
                {"false_when", "waiting"}}}
          )}}
    );
    const auto& question = std::get<DecisionRequest>(decision.input).questions[0];
    assert(std::get<NoulQuestion>(question.criteria).true_when == "ready");
    const auto video = request(
        "video", {{"feature", "video"}, {"prompt", "Mountain"}, {"resolution", "480p"}}
    );
    assert(std::get<VideoRequest>(video.input).short_edge == 480);
    assert(std::get<VideoRequest>(video.input).resolution_supplied);
    assert(!std::get<VideoRequest>(video.input).duration_supplied);
    for (const auto duration : {1u, 15u}) {
        const auto clip = request(
            "video",
            {{"feature", "video"}, {"prompt", "Mountain"}, {"duration", duration}}
        );
        assert(std::get<VideoRequest>(clip.input).duration == duration);
        assert(std::get<VideoRequest>(clip.input).duration_supplied);
    }
    rejects([&] {
        request(
            "video",
            {{"feature", "video"}, {"prompt", "Mountain"}, {"resolution", "1080p"}}
        );
    });
    rejects([&] {
        request(
            "video", {{"feature", "video"}, {"prompt", "Mountain"}, {"duration", 16}}
        );
    });
    const auto download =
        request("download", {{"feature", "models"}, {"model_id", "small"}}, "download");
    assert(std::get<DownloadRequest>(download.input).model == "small");
    rejects([&] { request("bad", {{"feature", "models"}, {"model_id", "small"}}); });
    rejects([&] {
        request("bad", {{"feature", "image"}, {"prompt", "Mountain"}}, "download");
    });
}

void codec_units() {
    assert(base64_encode("hello") == "aGVsbG8=");
    assert(base64_decode("aGVsbG8=") == "hello");
    assert(
        base64_decode(base64_encode(std::string("\0\xff", 2)))
        == std::string("\0\xff", 2)
    );
    for (const auto* invalid : {"a", "=AAA", "AA=A", "AA!A", "===="})
        rejects([&] { base64_decode(invalid); });
    Directory tmp;
    std::string wav = "RIFF";
    auto put = [&](std::uint32_t value, unsigned n) {
        for (unsigned i = 0; i < n; ++i)
            wav += char(value >> (8 * i));
    };
    put(40, 4);
    wav += "WAVEfmt ";
    put(16, 4);
    put(1, 2);
    put(1, 2);
    put(16000, 4);
    put(32000, 4);
    put(2, 2);
    put(16, 2);
    wav += "data";
    put(4, 4);
    put(32768, 2);
    put(16384, 2);
    const auto values = pcm_audio("data:audio/wav;base64," + base64_encode(wav));
    assert(values[0] == -1.0f && values[1] == 0.5f);
    wav.pop_back();
    rejects([&] { pcm_audio("data:audio/wav;base64," + base64_encode(wav)); });
    assert(!fs::exists(tmp.path / "invalid"));
}

void acquisition_units() {
    Directory tmp;
    const auto data = tmp.path / "data", root = tmp.path / "models";
    const auto manifest = local_manifest(
        tmp.path / "source", {{"weights.safetensors", "verified-bytes"}}
    );
    catalog(data, Json::array({entry("test", manifest)}));
    unsigned calls = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> progress;
    ModelStore store(
        root,
        data,
        [&](const auto&, const auto& write, const auto&, bool head, std::uint64_t) {
            assert(!head);
            ++calls;
            write("verified-bytes");
            return 14;
        },
        [&](const auto& model, auto received, auto total) {
            assert(model == "test");
            progress.emplace_back(received, total);
        }
    );
    const auto checkpoint = store.acquire("test", running);
    assert(calls == 1 && store.complete(store.entry("test")));
    assert(
        progress.front().first == 0 && progress.back().first == 14
        && progress.back().second == 14
    );
    assert(read_bytes(checkpoint / "weights.safetensors", 32) == "verified-bytes");
    assert(store.acquire("test", running) == checkpoint && calls == 1);
    assert(!store.allowed("../weights.safetensors", store.entry("test")));
    fs::remove_all(checkpoint);
    ModelStore corrupt(
        root,
        data,
        [](const auto&, const auto& write, const auto&, bool, std::uint64_t) {
            write("corrupt--bytes");
            return 14;
        }
    );
    rejects([&] { corrupt.acquire("test", running); });
    assert(fs::exists(root / ".test.partial") && !fs::exists(checkpoint));
    std::atomic_bool cancel{false};
    ModelStore cancelled(
        root,
        data,
        [&](const auto&, const auto& write, const auto&, bool, std::uint64_t) {
            write("verified-bytes");
            cancel = true;
            return 14;
        }
    );
    rejects([&] { cancelled.acquire("test", cancel); });
    assert(fs::exists(root / ".test.partial") && !fs::exists(checkpoint));
    cancel = false;
    assert(store.acquire("test", cancel) == checkpoint);  // writer lock was released
    fs::remove(checkpoint / "weights.safetensors");
    fs::create_symlink(
        tmp.path / "source/weights.safetensors", checkpoint / "weights.safetensors"
    );
    assert(!store.complete(store.entry("test")));
}

void recovery_units() {
    for (const bool cancellation : {false, true}) {
        Directory tmp;
        const auto data = tmp.path / "data", root = tmp.path / "models";
        const std::map<std::string, std::string> contents{
            {"a.safetensors", "verified"}, {"b.safetensors", "remaining"}
        };
        const auto files = local_manifest(tmp.path / "source", contents);
        catalog(data, Json::array({entry("test", files)}));
        std::atomic_bool cancel{false};
        ModelStore interrupted(
            root,
            data,
            [&](const std::string& url,
                const auto& write,
                const auto&,
                bool head,
                std::uint64_t offset) {
                assert(!head && offset == 0);
                const auto name = fs::path(url).filename().string();
                if (name == "a.safetensors") {
                    write(contents.at(name));
                    return 0;
                }
                write("rema");
                if (cancellation) {
                    cancel = true;
                    return 0;
                }
                throw RequestError("Unit connection interrupted", 503);
            }
        );
        rejects([&] { interrupted.acquire("test", cancel); });
        const auto stage = root / ".test.partial";
        assert(read_bytes(stage / "a.safetensors", 32) == "verified");
        assert(read_bytes(stage / "b.safetensors", 32) == "rema");
        assert(!fs::exists(stage / "complete.json"));
        assert(!interrupted.complete(interrupted.entry("test")));
        unsigned calls = 0;
        std::uint64_t progress = 0;
        // Metadata ordering is not part of artifact identity.
        auto reordered = files;
        std::reverse(reordered.begin(), reordered.end());
        catalog(data, Json::array({entry("test", reordered)}));
        // A new store simulates retry after a worker restart.
        ModelStore resumed(
            root,
            data,
            [&](const std::string& url,
                const auto& write,
                const auto&,
                bool head,
                std::uint64_t offset) {
                ++calls;
                assert(
                    !head && fs::path(url).filename() == "b.safetensors" && offset == 4
                );
                write("ining");
                return 5;
            },
            [&](const auto&, auto value, auto total) {
                assert(value <= total && value >= progress);
                progress = value;
            }
        );
        cancel = false;
        const auto path = resumed.acquire("test", cancel);
        assert(calls == 1 && progress == 17);
        assert(read_bytes(path / "b.safetensors", 32) == "remaining");
        assert(resumed.complete(resumed.entry("test")));
    }
}

struct PartialDownload {
    Directory tmp;
    fs::path data = tmp.path / "data", root = tmp.path / "models";
    fs::path stage = root / ".test.partial", artifact = stage / "weights.safetensors";
    Json model;

    PartialDownload() {
        const auto files =
            local_manifest(tmp.path / "source", {{"weights.safetensors", "verified"}});
        model = entry("test", files);
        catalog(data, Json::array({model}));
        ModelStore interrupted(
            root,
            data,
            [](const auto&, const auto& write, const auto&, bool, std::uint64_t)
                -> std::int64_t {
                write("veri");
                throw RequestError("Unit interruption", 503);
            }
        );
        rejects([&] { interrupted.acquire("test", running); });
        assert(read_bytes(artifact, 32) == "veri");
    }
};

void partial_identity_units() {
    for (const auto* field : {"revision", "repo_id", "digest", "size", "url"}) {
        PartialDownload partial;
        auto changed = partial.model;
        if (std::string(field) == "digest")
            changed["manifest"][0][field] = std::string(64, 'a');
        else if (std::string(field) == "size")
            changed["manifest"][0][field] = std::uint64_t(10);
        else if (std::string(field) == "url")
            changed["manifest"][0][field] = "https://unit.invalid/changed";
        else
            changed[field] = "changed";
        catalog(partial.data, Json::array({changed}));
        unsigned calls = 0;
        ModelStore mismatch(
            partial.root,
            partial.data,
            [&](const auto&, const auto&, const auto&, bool, std::uint64_t)
                -> std::int64_t {
                ++calls;
                return 0;
            }
        );
        rejects([&] { mismatch.acquire("test", running); });
        assert(calls == 0 && read_bytes(partial.artifact, 32) == "veri");
    }
}

void partial_path_units() {
    for (const bool symlink : {false, true}) {
        PartialDownload partial;
        fs::remove(partial.artifact);
        if (symlink)
            fs::create_symlink(
                partial.tmp.path / "source/weights.safetensors", partial.artifact
            );
        else
            bytes(partial.artifact, "oversized");
        unsigned calls = 0;
        ModelStore store(
            partial.root,
            partial.data,
            [&](const auto&, const auto&, const auto&, bool, std::uint64_t)
                -> std::int64_t {
                ++calls;
                return 0;
            }
        );
        rejects([&] { store.acquire("test", running); });
        assert(calls == 0);
        assert(
            read_bytes(partial.tmp.path / "source/weights.safetensors", 32)
            == "verified"
        );
    }
}

void partial_marker_units() {
    PartialDownload partial;
    const auto outside = partial.tmp.path / "unrelated.json";
    bytes(outside, "untouched");
    fs::create_symlink(outside, partial.stage / "complete.json");
    unsigned calls = 0;
    ModelStore store(
        partial.root,
        partial.data,
        [&](const auto&, const auto&, const auto&, bool, std::uint64_t)
            -> std::int64_t {
            ++calls;
            return 0;
        }
    );
    rejects([&] { store.acquire("test", running); });
    assert(calls == 0 && read_bytes(outside, 32) == "untouched");
}

void partial_integrity_units() {
    for (const bool complete : {false, true}) {
        PartialDownload partial;
        bytes(partial.artifact, complete ? "tampered" : "xxxx");
        unsigned calls = 0;
        ModelStore store(
            partial.root,
            partial.data,
            [&](
                const auto&, const auto& write, const auto&, bool, std::uint64_t offset
            ) {
                ++calls;
                assert(offset == 4);
                write("fied");
                return 4;
            }
        );
        rejects([&] { store.acquire("test", running); });
        assert(calls == (complete ? 0U : 1U) && !fs::exists(partial.artifact));
        assert(!store.complete(store.entry("test")));
    }
}

void partial_range_units() {
    for (const bool ignored : {false, true}) {
        PartialDownload partial;
        ModelStore store(
            partial.root,
            partial.data,
            [&](
                const auto&, const auto& write, const auto&, bool, std::uint64_t offset
            ) {
                assert(offset == 4);
                if (!ignored)
                    throw RequestError("Range unavailable", 503);
                write("verified");
                return 8;
            }
        );
        rejects([&] { store.acquire("test", running); });
        assert(read_bytes(partial.artifact, 32) == "veri");
        assert(!store.complete(store.entry("test")));
    }
}

void partial_initialization_units() {
    for (const auto* leftover : {"", "resume.json.worker.tmp", "weights.safetensors"}) {
        Directory tmp;
        const auto data = tmp.path / "data", root = tmp.path / "models";
        const auto stage = root / ".test.partial";
        const auto files =
            local_manifest(tmp.path / "source", {{"weights.safetensors", "verified"}});
        catalog(data, Json::array({entry("test", files)}));
        fs::create_directories(stage);
        if (*leftover)
            bytes(stage / leftover, "unfinished");
        unsigned calls = 0;
        ModelStore store(
            root,
            data,
            [&](
                const auto&, const auto& write, const auto&, bool, std::uint64_t offset
            ) {
                ++calls;
                assert(offset == 0);
                write("verified");
                return 8;
            }
        );
        if (std::string(leftover) == "weights.safetensors") {
            rejects([&] { store.acquire("test", running); });
            assert(calls == 0 && read_bytes(stage / leftover, 32) == "unfinished");
        } else {
            assert(fs::exists(store.acquire("test", running) / "complete.json"));
            assert(calls == 1);
        }
    }
}

void single_file_acquisition_units() {
    Directory tmp;
    const auto data = tmp.path / "data", root = tmp.path / "models";
    const auto files =
        local_manifest(tmp.path / "source", {{"test.pt", "metadata-only"}});
    auto model = entry("whisper-test", files);
    model["layout"] = "single_file";
    model["asset_name"] = "test.pt";
    model["asset_url"] = "https://unit.invalid/test.pt";
    model["revision"] = files[0].at("digest");
    catalog(data, Json::array({model}));
    unsigned bodies = 0, heads = 0;
    bool missing_size = true, corrupt = false;
    ModelStore store(
        root,
        data,
        [&](const auto&, const auto& write, const auto&, bool head, std::uint64_t) {
            if (head) {
                ++heads;
                return missing_size ? -1 : 13;
            }
            ++bodies;
            write(corrupt ? "corrupt-bytes" : "metadata-only");
            return 13;
        }
    );
    rejects([&] { store.acquire("whisper-test", running); });
    assert(heads == 1 && bodies == 0);
    missing_size = false;
    corrupt = true;
    rejects([&] { store.acquire("whisper-test", running); });
    assert(fs::exists(root / ".whisper-test.partial"));
    corrupt = false;
    const auto checkpoint = store.acquire("whisper-test", running);
    assert(
        store.complete(CatalogEntry::parse(model))
        && read_bytes(checkpoint / "test.pt", 32) == "metadata-only"
    );
    auto marker = read_document(checkpoint / "complete.json");
    marker["files"][0]["digest"] = std::string(64, 'a');
    write_document(checkpoint / "complete.json", marker);
    assert(!store.complete(CatalogEntry::parse(model)));
}

void upstream_manifest_units() {
    Directory tmp;
    const auto data = tmp.path / "data", root = tmp.path / "models";
    std::map<std::string, std::string> contents{
        {"config.json", "{}"},
        {"tokenizer_config.json", "{}"},
        {"model.safetensors.index.json",
         R"({"weight_map":{"weight":"model.safetensors"}})"},
        {"model.safetensors", "metadata-only"}
    };
    const auto files = local_manifest(tmp.path / "source", contents);
    auto model = entry("test", files);
    model["layout"] = "root";
    catalog(data, Json::array({model}));
    auto document = read_document(data / "catalog.json");
    document["assets"] = {
        "config.json", "tokenizer_config.json", "model.safetensors.index.json"
    };
    write_document(data / "catalog.json", document);
    Json siblings = Json::array();
    for (const auto& file : files)
        siblings.push_back(
            {{"rfilename", file.at("name")},
             {"lfs", {{"size", file.at("size")}, {"sha256", file.at("digest")}}}}
        );
    // Exercise git-blob integrity as well as LFS SHA256 without fetching weights.
    siblings[0].erase("lfs");
    siblings[0]["size"] = 2;
    siblings[0]["blobId"] = "9e26dfeeb6e641a33dae4961196235bdb965b21b";
    Json upstream{{"sha", "wrong-revision"}, {"siblings", siblings}};
    unsigned bodies = 0;
    std::vector<std::string> fetched;
    bool interrupt_config = true;
    ModelStore store(
        root,
        data,
        [&](const std::string& url,
            const auto& write,
            const auto&,
            bool head,
            std::uint64_t offset) {
            assert(!head);
            if (url.find("/api/models/") != std::string::npos)
                write(upstream.dump());
            else {
                ++bodies;
                fetched.push_back(fs::path(url).filename().string());
                const auto name = fetched.back();
                if (name == "config.json" && interrupt_config) {
                    interrupt_config = false;
                    write("{");
                    throw RequestError("Unit connection interrupted", 503);
                }
                write(contents.at(name).substr(offset));
            }
            return 0;
        }
    );
    rejects([&] { store.acquire("test", running); });
    assert(bodies == 0);
    upstream["sha"] = "revision";
    upstream["siblings"].push_back(siblings[0]);
    rejects([&] { store.acquire("test", running); });
    assert(bodies == 0);
    upstream["siblings"] = siblings;
    rejects([&] { store.acquire("test", running); });
    assert(read_bytes(root / ".test.partial/config.json", 32) == "{");
    const auto checkpoint = store.acquire("test", running);
    assert(bodies == 5 && store.complete(CatalogEntry::parse(model)));
    assert(fetched.front() == "model.safetensors.index.json");
    assert(store.acquire("test", running) == checkpoint && bodies == 5);
    fs::remove_all(checkpoint);
    // Authenticate and validate index metadata before any tensor transfer.
    for (const Json& map :
         {Json::object(),
          Json::array({"model.safetensors"}),
          Json("model.safetensors"),
          Json{{"weight", 1}},
          Json{{"weight", "missing.safetensors"}},
          Json{{"weight", "../outside.safetensors"}},
          Json{{"weight", "/outside.safetensors"}},
          Json{{"weight", "config.json"}}}) {
        contents["model.safetensors.index.json"] = Json{{"weight_map", map}}.dump();
        const auto changed = local_manifest(tmp.path / "source", contents);
        for (std::size_t i = 0; i < changed.size(); ++i)
            if (changed[i].at("name") == "model.safetensors.index.json")
                upstream["siblings"][i]["lfs"] = {
                    {"size", changed[i].at("size")}, {"sha256", changed[i].at("digest")}
                };
        fetched.clear();
        rejects([&] { store.acquire("test", running); });
        assert(fetched == std::vector<std::string>{"model.safetensors.index.json"});
        assert(!fs::exists(checkpoint) && fs::exists(root / ".test.partial"));
        // Each changed upstream index is an independent fixture.
        fs::remove_all(root / ".test.partial");
    }
    for (auto& item : upstream["siblings"])
        if (item["rfilename"] == "model.safetensors.index.json")
            item["lfs"]["size"] = 32 * 1024 * 1024 + 1;
    fetched.clear();
    rejects([&] { store.acquire("test", running); });
    assert(fetched.empty());

    for (const bool valid : {false, true}) {
        contents = {
            {"transformer/model.safetensors", "payload"},
            {"text_encoder/model.safetensors", "payload"},
            {"transformer/model.safetensors.index.json",
             R"({"weight_map":{"weight":"model.safetensors"}})"},
            {"text_encoder/model.safetensors.index.json",
             Json{
                 {"weight_map",
                  {{"weight", valid ? "model.safetensors" : "missing.safetensors"}}}
             }.dump()}
        };
        auto component_files = local_manifest(tmp.path / "source", contents);
        // Interleave payloads and indexes to exercise download ordering.
        std::reverse(component_files.begin(), component_files.end());
        const auto component = entry("components", component_files);
        catalog(data, Json::array({component}));
        fetched.clear();
        ModelStore components(
            root,
            data,
            [&](const std::string& url,
                const auto& write,
                const auto&,
                bool head,
                std::uint64_t) {
                assert(!head);
                const auto name =
                    url.substr(std::string("https://unit.invalid/").size());
                fetched.push_back(name);
                write(contents.at(name));
                return 0;
            }
        );
        if (valid) {
            const auto path = components.acquire("components", running);
            assert(components.complete(components.entry("components")));
            assert(components.acquire("components", running) == path);
            assert(fetched.size() == 4);
            // Composite completion comparison requires the original manifest order.
            assert(
                read_document(path / "complete.json").at("files") == component_files
            );
        } else {
            rejects([&] { components.acquire("components", running); });
            assert(fetched.size() == 2);
            assert(fs::exists(root / ".components.partial"));
            fs::remove_all(root / ".components.partial");
        }
        assert(fetched[0] == "transformer/model.safetensors.index.json");
        assert(fetched[1] == "text_encoder/model.safetensors.index.json");
    }
}

void job_units() {
    Directory tmp;
    const auto data = tmp.path / "data", root = tmp.path / "models",
               jobs = tmp.path / "jobs";
    const auto config = Json::object().dump();
    Json models = Json::array();
    for (const auto* id : {"qwen-image-2.1-turbo", "laya", "h3-fl2va-int8-turbo"}) {
        const auto dir = root / (std::string(id) + "-revision");
        const auto manifest = local_manifest(dir, {{"config.json", config}});
        models.push_back(entry(id, manifest));
        write_document(
            dir / "complete.json", {{"revision", "revision"}, {"files", manifest}}
        );
    }
    catalog(data, models);
    // Old selection files are neither read nor deleted by the tierless store.
    write_document(root / "tier.json", {{"tier", "go"}});
    write_document(root / "context.json", {{"small", 4096}});
    const auto legacy_tier = read_bytes(root / "tier.json", 4096);
    const auto legacy_context = read_bytes(root / "context.json", 4096);
    JobWorkspace files(jobs);
    ModelStore store(
        root,
        data,
        [](const auto&, const auto&, const auto&, bool, std::uint64_t) -> std::int64_t {
            throw RequestError("Unexpected network access in preparation unit");
        }
    );
    const auto initial = store.status();
    assert(initial.at("selected_model_id") == "lfm3-1.7b-bf16");
    assert(!initial.contains("tiers") && !initial.contains("selected_tier"));
    std::string oversized_prompt;
    for (unsigned index = 0; index < 3000; ++index)
        oversized_prompt += "狐";
    auto oversized_image = request(
        "oversized-image",
        {{"feature", "image"}, {"prompt", oversized_prompt}, {"count", 1}}
    );
    bool prompt_rejected = false;
    try {
        prepare_task(oversized_image, store, files, running);
    } catch (const RequestError& error) {
        prompt_rejected = error.status == 422;
    }
    assert(prompt_rejected);
    files.cleanup(oversized_image.id);
    auto defaulted = request(
        "defaulted-image", {{"feature", "image"}, {"prompt", "mountain"}, {"count", 1}}
    );
    prepare_task(defaulted, store, files, running);
    assert(
        defaulted.model == "qwen-image-2.1-turbo"
        && std::get<ImageRequest>(defaulted.input).steps == 8
    );
    files.cleanup(defaulted.id);
    auto video =
        request("defaulted-video", {{"feature", "video"}, {"prompt", "mountain"}});
    prepare_task(video, store, files, running);
    assert(std::get<VideoRequest>(video.input).duration == 5);
    assert(std::get<VideoRequest>(video.input).short_edge == 480);
    files.cleanup(video.id);
    auto explicit_video = request(
        "explicit-video",
        {{"feature", "video"},
         {"prompt", "mountain"},
         {"duration", 15},
         {"resolution", "768p"}}
    );
    prepare_task(explicit_video, store, files, running);
    assert(std::get<VideoRequest>(explicit_video.input).duration == 15);
    assert(std::get<VideoRequest>(explicit_video.input).short_edge == 768);
    files.cleanup(explicit_video.id);
    // Fixed media assignments do not require any tier selection.
    auto untiered = request(
        "untiered-image", {{"feature", "image"}, {"prompt", "mountain"}, {"count", 1}}
    );
    prepare_task(untiered, store, files, running);
    assert(untiered.model == "qwen-image-2.1-turbo");
    files.cleanup(untiered.id);
    // Turbo remains the fixed image model.
    rejects([&] {
        request(
            "image-selection",
            {{"feature", "image_selection"}, {"model_id", "qwen-image-2.1-turbo"}},
            "configure"
        );
    });
    auto image = request(
        "image",
        {{"feature", "image"},
         {"prompt", "mountain"},
         {"count", 2},
         {"steps", 20},
         {"seed", UINT64_MAX}}
    );
    prepare_task(image, store, files, running);
    assert(
        image.model == "qwen-image-2.1-turbo"
        && std::get<ImageRequest>(image.input).width == 2048
        && std::get<ImageRequest>(image.input).steps == 8
    );
    bytes(jobs / "image/one.png", "image-one");
    bytes(jobs / "image/two.png", "image-two");
    const auto packaged =
        present_image(image, {jobs / "image/one.png", jobs / "image/two.png"})
            .at("image");
    assert(packaged.at("seeds")[0] == UINT64_MAX && packaged.at("seeds")[1] == 0);
    assert(
        packaged.at("images_base64")[0].get<std::string>() == base64_encode("image-one")
    );
    rejects([&] { present_image(image, {"/outside.png", "/outside2.png"}); });
    files.cleanup("image");
    assert(!fs::exists(jobs / "image"));
    for (unsigned steps : {2u, 50u, 100u}) {
        auto turbo = request(
            "turbo",
            {{"feature", "image"},
             {"model_id", "qwen-image-2.1-turbo"},
             {"prompt", "mountain"},
             {"count", 1},
             {"steps", steps},
             {"seed", 7}}
        );
        prepare_task(turbo, store, files, running);
        const auto& input = std::get<ImageRequest>(turbo.input);
        assert(turbo.model == "qwen-image-2.1-turbo" && input.steps == 8);
        assert(input.width == 2048 && input.height == 2048 && input.seed == 7);
        assert(
            turbo.configuration.checkpoint == root / "qwen-image-2.1-turbo-revision"
        );
        bytes(jobs / "turbo/one.png", "image-one");
        const auto meta = present_image(turbo, {jobs / "turbo/one.png"})
                              .at("image")
                              .at("meta")
                              .get<std::string>();
        assert(meta.find("8 steps") != std::string::npos);
        files.cleanup("turbo");
    }
    const std::string small_source_png =
        "data:image/"
        "png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAACAAAABACAYAAAB7jnWuAAAAP0lEQVR4nO3OMQEAAAjDsIF/"
        "z0MGT2qgmbbNY/"
        "s5BwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEiSA61CBHwO9WIDAAAAAElFTkSuQmCC";
    const std::string source_png =
        "data:image/"
        "png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAADHUlEQVR4nO3UMQEAIAzAMMC/"
        "5yFjRxMFvXpnZg6Q9LYDgD0GAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAG"
        "EGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEG"
        "AGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAG"
        "EGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEG"
        "AGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAG"
        "EGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEG"
        "AGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAG"
        "EGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEG"
        "AGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAG"
        "EGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEG"
        "AGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAG"
        "EGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEG"
        "AGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAGEGAG"
        "EGAGEGAGEGAGEGAGEGAGEGAGEfHrAF/LZ/lzEAAAAASUVORK5CYII=";
    for (bool explicit_size : {false, true}) {
        Json body = {
            {"feature", "image"},
            {"prompt", "fox jumping"},
            {"image", source_png},
            {"mode", "reference"},
            {"count", 1}
        };
        if (explicit_size) {
            body["width"] = 384;
            body["height"] = 416;
        }
        auto reference = request("reference-image", body);
        prepare_task(reference, store, files, running);
        const auto& input = std::get<ImageRequest>(reference.input);
        assert(
            input.reference && input.source_width == 256 && input.source_height == 256
        );
        assert(input.width == (explicit_size ? 384u : 256u));
        assert(input.height == (explicit_size ? 416u : 256u));
        assert(input.steps == 8);
        bytes(jobs / "reference-image/one.png", "image-one");
        const auto meta = present_image(reference, {jobs / "reference-image/one.png"})
                              .at("image")
                              .at("meta")
                              .get<std::string>();
        assert(meta.find("8 steps") != std::string::npos);
        files.cleanup("reference-image");
    }
    rejects([&] {
        auto small = request(
            "small-reference",
            {{"feature", "image"},
             {"prompt", "fox"},
             {"image", small_source_png},
             {"mode", "reference"}}
        );
        prepare_task(small, store, files, running);
    });
    for (const auto& invalid : std::vector<Json>{
             {{"mode", "unknown"}},
             {{"mode", "reference"}, {"strength", .65}},
             {{"mode", "reference"}, {"mask", source_png}},
             {{"mode", "reference"}, {"width", 384}},
             {{"mode", "reference"}, {"width", 0}, {"height", 416}},
             {{"mode", "reference"}, {"width", 385}, {"height", 416}},
             {{"mode", "reference"}, {"width", 3072}, {"height", 3072}},
             {{"width", 384}, {"height", 416}}
         }) {
        Json body = {{"feature", "image"}, {"prompt", "fox"}, {"image", source_png}};
        body.update(invalid);
        rejects([&] { request("invalid-reference", body); });
    }
    rejects([&] {
        request(
            "missing-reference",
            {{"feature", "image"}, {"prompt", "fox"}, {"mode", "reference"}}
        );
    });
    auto unsupported = request(
        "unsupported-image",
        {{"feature", "image"}, {"model_id", "small"}, {"prompt", "mountain"}}
    );
    rejects([&] { prepare_task(unsupported, store, files, running); });
    std::atomic_bool cancel{true};
    rejects([&] { prepare_task(image, store, files, cancel); });
    assert(!fs::exists(jobs / "image"));
    auto decision = request(
        "decisions",
        {{"feature", "decisions"},
         {"state", "example"},
         {"questions",
          Json::array(
              {{{"key", "pick"},
                {"type", "Choice"},
                {"instructions", "Pick"},
                {"options", Json::array({{{"key", "yes"}, {"description", "yes"}}})}}}
          )}}
    );
    prepare_task(decision, store, files, running);
    const auto answers = present_decisions(
        {{"pick", 0.75, ChoiceResult{"yes", {{"yes", 0.75}, {"no", 0.25}}}},
         {"score", 0.6, ScoreResult{0.6, {{"0", 0.4}, {"1", 0.6}}}},
         {"criterion", 0.8, NoulResult{0.8}}}
    );
    assert(answers[0].at("type") == "Choice" && answers[0].at("value") == "yes");
    assert(answers[1].at("type") == "Score" && answers[1].at("value") == 0.6);
    assert(answers[2].at("type") == "Noul" && answers[2].at("probabilities").is_null());
    files.cleanup("decisions");
    assert(read_bytes(root / "tier.json", 4096) == legacy_tier);
    assert(read_bytes(root / "context.json", 4096) == legacy_context);
}

void write_archive(
    const fs::path& path, const std::map<std::string, std::string>& entries
) {
    std::unique_ptr<struct archive, decltype(&archive_write_free)> archive(
        archive_write_new(), archive_write_free
    );
    assert(archive_write_set_format_zip(archive.get()) == ARCHIVE_OK);
    assert(archive_write_open_filename(archive.get(), path.c_str()) == ARCHIVE_OK);
    for (const auto& [name, content] : entries) {
        std::unique_ptr<struct archive_entry, decltype(&archive_entry_free)> item(
            archive_entry_new(), archive_entry_free
        );
        archive_entry_set_pathname(item.get(), name.c_str());
        archive_entry_set_size(item.get(), content.size());
        archive_entry_set_filetype(item.get(), AE_IFREG);
        archive_entry_set_perm(item.get(), 0600);
        assert(archive_write_header(archive.get(), item.get()) == ARCHIVE_OK);
        assert(
            archive_write_data(archive.get(), content.data(), content.size())
            == static_cast<la_ssize_t>(content.size())
        );
    }
    assert(archive_write_close(archive.get()) == ARCHIVE_OK);
}

std::string pickle_string(std::string_view value) {
    std::string result = "X";
    for (unsigned i = 0; i < 4; ++i)
        result += char(value.size() >> (8 * i));
    result += value;
    return result;
}

void whisper_conversion_units() {
    Directory tmp;
    std::string pickle = "\x80\x02}(";
    pickle += pickle_string("dims") + "}(";
    for (const auto* name :
         {"n_mels",
          "n_audio_ctx",
          "n_audio_state",
          "n_audio_head",
          "n_audio_layer",
          "n_vocab",
          "n_text_ctx",
          "n_text_state",
          "n_text_head",
          "n_text_layer"})
        pickle += pickle_string(name) + "K\x02";
    pickle += "u" + pickle_string("model_state_dict") + "ccollections\nOrderedDict\n)R";
    pickle += pickle_string("test.weight") + "ctorch._utils\n_rebuild_tensor_v2\n(("
              + pickle_string("storage") + "ctorch\nFloatStorage\n" + pickle_string("0")
              + pickle_string("cpu") + "K\x04tQ";
    pickle += std::string("K\0", 2)
              + "K\x02K\x02\x86K\x01K\x02\x86\x89"
                "ccollections\nOrderedDict\n)RtRsu.";
    std::array<float, 4> values = {1, 2, 3, 4};
    const std::string raw(reinterpret_cast<const char*>(values.data()), sizeof(values));
    const auto source = tmp.path / "source.pt";
    write_archive(source, {{"archive/data.pkl", pickle}, {"archive/data/0", raw}});
    export_whisper(
        source, file_digest(source, running), tmp.path / "converted", running
    );
    const auto bytes = read_bytes(tmp.path / "converted/model.safetensors", 4096);
    std::uint64_t header_size = 0;
    std::memcpy(&header_size, bytes.data(), 8);
    const auto header = Json::parse(bytes.substr(8, header_size));
    assert(header.at("test.weight").at("dtype") == "F32");
    std::array<float, 4> result{};
    std::memcpy(result.data(), bytes.data() + 8 + header_size, sizeof(result));
    assert((result == std::array<float, 4>{1, 3, 2, 4}));
    rejects([&] {
        export_whisper(source, std::string(64, '0'), tmp.path / "invalid", running);
    });
    assert(!fs::exists(tmp.path / "invalid"));
    write_archive(
        source,
        {{"archive/data.pkl",
          "\x80\x02"
          "cos\nsystem\n)R."}}
    );
    rejects([&] {
        export_whisper(
            source, file_digest(source, running), tmp.path / "unsafe", running
        );
    });
    assert(!fs::exists(tmp.path / "unsafe"));
}

void tokenizer_asset_units() {
    Directory tmp;
    for (const auto vocabulary : {51864u, 51865u, 51866u}) {
        const auto destination = tmp.path / std::to_string(vocabulary);
        export_whisper_assets(
            vocabulary, worker_data_directory() / "assets", destination, running
        );
        const auto metadata = read_document(destination / "assets.json");
        assert(metadata.at("vocabulary") == vocabulary);
        assert(metadata.at("prompt").size() == (vocabulary == 51864 ? 2 : 4));
        // Read the exported metadata, without constructing or running a model.
        std::ifstream tokens(destination / "english.tokens", std::ios::binary);
        tokens.seekg(16);  // magic, vocabulary, EOS
        const auto read_number = [&] {
            std::array<unsigned char, 4> bytes{};
            tokens.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
            assert(tokens.good());
            return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8)
                   | (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
        };
        auto count = read_number();
        tokens.seekg(count * 4, std::ios::cur);  // prompt
        count = read_number();
        for (std::uint32_t i = 0; i < count; ++i)
            assert(read_number() != metadata.at("eos").get<std::uint32_t>());
        count = read_number();
        bool first_step_suppresses_eos = false;
        for (std::uint32_t i = 0; i < count; ++i)
            first_step_suppresses_eos |=
                read_number() == metadata.at("eos").get<std::uint32_t>();
        assert(first_step_suppresses_eos);

        assert(fs::file_size(destination / "mel-80.f32") == 80 * 201 * 4);
        assert(fs::file_size(destination / "mel-128.f32") == 128 * 201 * 4);
        assert(
            read_bytes(destination / "english.tokens", 8 * 1024 * 1024).substr(0, 8)
            == "KDWVOC01"
        );
    }
}
}  // namespace

int main() {
    manifest_model_units();
    pinned_catalog_units();
    shared_settings_units();
    whisper_formatting_setting_units();
    whisper_formatting_request_units();
    whisper_formatting_response_units();
    checkpoint_catalog_units();
    typed_request_units();
    codec_units();
    acquisition_units();
    recovery_units();
    partial_identity_units();
    partial_path_units();
    partial_marker_units();
    partial_integrity_units();
    partial_range_units();
    partial_initialization_units();
    single_file_acquisition_units();
    upstream_manifest_units();
    job_units();
    whisper_conversion_units();
    tokenizer_asset_units();
    std::cout << "C++ preparation units passed; no network or model execution.\n";
}
