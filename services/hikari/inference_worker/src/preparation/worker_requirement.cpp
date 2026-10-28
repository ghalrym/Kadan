#include "kadan/h3_memory.hpp"
#include "kadan/preparation.hpp"
#include "kadan/worker_compute.hpp"

namespace kadan::serving {
namespace {
constexpr Bytes gib = 1024ULL * 1024 * 1024;
// Media engines admit a device only with this much headroom (media_plan).
constexpr Bytes device_headroom = compute_context_bytes + 2 * gib;
// Engine admission floors, from each engine's load().
constexpr Bytes image_host_floor = 16 * gib;
constexpr Bytes video_device_floor = 3 * gib;
constexpr Bytes decision_envelope = 3 * gib;
// Video prompts beyond this length no longer grow the text stage.
constexpr std::size_t max_prompt_bytes = 32000;

std::string default_model(Workload workload, const ModelStore& store) {
    switch (workload) {
    case Workload::llm:
        throw RequestError("Language inference belongs to the LFM3 worker.", 422);
    case Workload::decision:
        return "laya";
    case Workload::formatting:
        throw RequestError("Transcript formatting belongs to the LFM3 worker.", 422);
    case Workload::speech:
        return "whisper-" + store.whisper_selected();
    case Workload::image:
        return "qwen-image-2.1-turbo";
    case Workload::tts:
        throw RequestError("Speech synthesis belongs to the LFM3 Audio worker.", 422);
    case Workload::video:
        return "h3-fl2va-int8-turbo";
    }
    throw std::invalid_argument("workload");
}

// Device amounts are for the whole workload on one GPU. Each further GPU it
// spreads across needs another `device_floor` of its own (its compute context
// and admission headroom); the rest divides between them.
Json footprint(
    const std::string& model,
    Bytes host_min,
    Bytes host_pref,
    Bytes device_min,
    Bytes device_pref,
    Bytes device_floor
) {
    return {
        {"model", model},
        {"minimum", {host_min, device_min}},
        {"preferred", {host_pref, device_pref}},
        {"device_floor", device_floor}
    };
}
}  // namespace

Json worker_requirement(
    Workload workload,
    const std::optional<std::string>& requested,
    const ModelStore& store,
    const std::optional<Json>&
) {
    // Transcription requests name a Whisper size; its catalog entry is prefixed.
    const auto model = workload == Workload::speech && requested
                           ? "whisper-" + store.whisper_model(*requested)
                       : requested ? *requested
                                   : default_model(workload, store);
    const auto& entry = store.entry(model);
    const Bytes weights = entry.estimated_bytes;
    // Minimums mirror each engine's own admission floor, so a granted budget
    // never leaves an engine waiting on memory it was promised. Preferred
    // amounts let an engine keep its whole checkpoint resident.
    switch (workload) {
    case Workload::decision:
        // Laya runs on the host only, inside a fixed envelope.
        return footprint(model, decision_envelope, decision_envelope, 0, 0, 0);
    case Workload::speech:
        return footprint(
            model,
            gib,
            weights + gib,
            device_headroom,
            weights + device_headroom,
            device_headroom
        );
    case Workload::image:
        return footprint(
            model,
            image_host_floor,
            weights + image_host_floor,
            device_headroom,
            weights + device_headroom,
            device_headroom
        );
    case Workload::video: {
        // The largest clip the API accepts bounds the host execution stage;
        // spare host memory beyond it becomes the engine's weight cache.
        const auto largest = video::h3_api_profile(768, "16:9", 15);
        const Bytes stage =
            std::max(4 * gib, video::h3_request_memory(largest, max_prompt_bytes));
        return footprint(
            model,
            stage,
            stage + 2 * weights,
            video_device_floor,
            weights + video_device_floor,
            video_device_floor
        );
    }
    case Workload::llm:
    case Workload::tts:
    case Workload::formatting:
        throw RequestError("This inference type belongs to an LFM3 worker.", 422);
    }
    throw std::invalid_argument("workload");
}
}  // namespace kadan::serving
