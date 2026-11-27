// Unified multi-workload C++ worker binary for Hikari.
// Dispatches one isolated workload per process (--workload=<kind>) so the
// desktop supervisor or container orchestrator can toggle, drain, and restart
// capabilities at runtime without per-workload container images.
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "kadan/worker_runtime.hpp"
#ifdef KADAN_HAS_IMAGE_ENGINE
#include "kadan/image_edit.hpp"
#endif

namespace {
using namespace kadan::serving;

constexpr const char* kWorkerVersion = "0.1.0";

struct CompiledWorkload {
    const char* name;
    const char* label;
    bool uses_gpu;
};

std::vector<CompiledWorkload> compiled_workloads() {
    std::vector<CompiledWorkload> out;
#ifdef KADAN_WORKER_CUDA
    out.push_back({"llm", "Language, Vision & Decisions (Inkling 2 Mini)", true});
#endif
    out.push_back({"lfm3_audio", "Voice Calls (LFM3 Audio 2B)", true});
    out.push_back({"formatting", "Speech Formatting (S1-mini CPU)", false});
#ifdef KADAN_WORKER_CUDA
    out.push_back({"speech", "Speech to Text (Whisper)", true});
#ifdef KADAN_HAS_TTS_ENGINE
    out.push_back({"tts", "Text to Speech (Qwen3-TTS)", true});
#endif
#ifdef KADAN_HAS_IMAGE_ENGINE
    out.push_back({"image", "Image Generation (Qwen-Image Turbo)", true});
#endif
#ifdef KADAN_HAS_VIDEO_ENGINE
    out.push_back({"video", "Video Generation (MiniMax H3 Turbo)", true});
#endif
#endif
    return out;
}

void print_help() {
    std::cout
        << "kadan-worker " << kWorkerVersion << "\n"
        << "Usage: kadan-worker --workload=<name> [options]\n\n"
        << "Options:\n"
        << "  --workload=<name>     Workload to run (required unless listing)\n"
        << "  --api-url=<ws-url>    Hikari API WebSocket URL (or HIKARI_API_URL)\n"
        << "  --token=<token>       Worker API token (or HIKARI_WORKER_TOKEN)\n"
        << "  --model-dir=<path>    Model storage directory (or KADAN_MODEL_DIR)\n"
        << "  --machine-name=<str>  Display name for this computer\n"
        << "  --instance=<str>      Instance identifier for this worker\n"
        << "  --gpu=<index>         Pin to a specific GPU index\n"
        << "  --cpu-threads=<n>     CPU thread count hint\n"
        << "  --config=<json>       Load defaults from a JSON config file\n"
        << "  --watch-stdin         Exit cleanly when stdin closes\n"
        << "  --list-workloads      Print compiled workloads as JSON and exit\n"
        << "  --version             Print version and exit\n";
}

std::optional<WorkerProgram> build_program(std::string_view workload) {
    if (workload == "lfm3_llm") {
        return WorkerProgram{
            kadan::Workload::lfm3_llm,
#ifdef KADAN_WORKER_CUDA
            true,
#else
            false,
#endif
            [](WorkerEnvironment& env) {
                auto& ch = env.channel;
                return lfm3_text_engine(
                    env.resources,
                    [&ch](const char* phase, std::size_t value) {
                        ch.progress(phase, value);
                    },
                    [&ch](const std::string& text) { ch.content(text); }
                );
            }
        };
    }
    if (workload == "lfm3_vision") {
        return WorkerProgram{
            kadan::Workload::lfm3_vision,
#ifdef KADAN_WORKER_CUDA
            true,
#else
            false,
#endif
            [](WorkerEnvironment& env) {
                auto& ch = env.channel;
                return lfm3_vision_engine(
                    env.resources,
                    [&ch](const char* phase, std::size_t value) {
                        ch.progress(phase, value);
                    }
                );
            }
        };
    }
    if (workload == "lfm3_audio") {
        return WorkerProgram{
            kadan::Workload::lfm3_audio,
#ifdef KADAN_WORKER_CUDA
            true,
#else
            false,
#endif
            [](WorkerEnvironment& env) {
                auto& ch = env.channel;
                return lfm3_audio_engine(
                    env.resources,
                    [&ch](const char* phase, std::size_t value) {
                        ch.progress(phase, value);
                    },
                    [&ch](const Json& event) { ch.send(event); }
                );
            }
        };
    }
    if (workload == "formatting") {
        return WorkerProgram{
            kadan::Workload::formatting,
            false,
            [](WorkerEnvironment& env) {
                auto& ch = env.channel;
                return transcription_formatting_engine(
                    env.resources,
                    [&ch](const char* phase, std::size_t value) {
                        ch.progress(phase, value);
                    }
                );
            }
        };
    }
#ifdef KADAN_WORKER_CUDA
    if (workload == "llm") {
        return WorkerProgram{
            kadan::Workload::llm,
            true,
            [](WorkerEnvironment& env) {
                auto& ch = env.channel;
                return chat_engine(
                    env.resources,
                    [&ch](const char* phase, std::size_t value) {
                        ch.progress(phase, value);
                    },
                    [&ch](const std::string& text) { ch.content(text); }
                );
            }
        };
    }
    if (workload == "speech") {
        return WorkerProgram{
            kadan::Workload::speech,
            true,
            [](WorkerEnvironment& env) {
                auto& ch = env.channel;
                return whisper_engine(
                    env.resources,
                    [&ch](const char* phase, std::size_t value) {
                        ch.progress(phase, value);
                    }
                );
            }
        };
    }
#ifdef KADAN_HAS_TTS_ENGINE
    if (workload == "tts") {
        return WorkerProgram{
            kadan::Workload::tts,
            true,
            [](WorkerEnvironment& env) {
                auto& ch = env.channel;
                return speech_engine(
                    env.resources,
                    [&ch](const char* phase, std::size_t value) {
                        ch.progress(phase, value);
                    }
                );
            }
        };
    }
#endif
#ifdef KADAN_HAS_IMAGE_ENGINE
    if (workload == "image") {
        return WorkerProgram{
            kadan::Workload::image,
            true,
            [](WorkerEnvironment& env) {
                auto& ch = env.channel;
                return image_engine(
                    env.resources,
                    [&ch](const char* phase, std::size_t value) {
                        ch.progress(phase, value);
                    },
                    [&ch](
                        const std::string& id, const char* phase, std::size_t value
                    ) {
                        ch.send(
                            {{"id", id},
                             {"state", "progress"},
                             {"phase", phase},
                             {"value", value}}
                        );
                    },
                    ch.available
                );
            },
            [](const InferenceRequest& request, Json& prepared) {
                const auto* image = std::get_if<ImageRequest>(&request.input);
                if (image)
                    prepared["total"] =
                        image->source && !image->reference
                            ? kadan::image::edit_steps(image->steps, image->strength)
                            : image->steps;
            }
        };
    }
#endif
#ifdef KADAN_HAS_VIDEO_ENGINE
    if (workload == "video") {
        return WorkerProgram{
            kadan::Workload::video,
            true,
            [](WorkerEnvironment& env) {
                auto& ch = env.channel;
                return video_engine(
                    env.resources,
                    [&ch](const char* phase, std::size_t value) {
                        ch.progress(phase, value);
                    }
                );
            }
        };
    }
#endif
#endif
    return std::nullopt;
}
}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        }
        if (arg == "--version") {
            std::cout << kWorkerVersion << "\n";
            return 0;
        }
        if (arg == "--list-workloads") {
            Json list = Json::array();
            for (const auto& w : compiled_workloads()) {
                list.push_back(
                    {{"id", w.name}, {"label", w.label}, {"uses_gpu", w.uses_gpu}}
                );
            }
            Json out = {
                {"version", kWorkerVersion},
#ifdef KADAN_WORKER_CUDA
                {"cuda", true},
#else
                {"cuda", false},
#endif
                {"workloads", std::move(list)}
            };
            std::cout << out.dump() << "\n";
            return 0;
        }
    }

    try {
        WorkerConfig config = parse_worker_config(argc, argv);
        if (config.workload.empty()) {
            if (const char* env_w = std::getenv("KADAN_WORKER_WORKLOAD");
                env_w && *env_w)
                config.workload = env_w;
        }
        if (config.workload.empty()) {
            std::cerr << "kadan-worker: --workload is required (use --list-workloads to see available workloads)\n";
            return 2;
        }
        auto program = build_program(config.workload);
        if (!program) {
            std::cerr << "kadan-worker: unsupported or uncompiled workload '"
                      << config.workload << "'\n";
            return 2;
        }
        return run_worker(*program, config);
    } catch (const std::exception& error) {
        std::cerr << "kadan-worker: " << error.what() << "\n";
        return 1;
    }
}
