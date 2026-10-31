#include "kadan/gguf_checkpoint.hpp"
#include "kadan/gguf_text.hpp"
#include "kadan/h3_memory.hpp"
#include "kadan/preparation.hpp"
#include "kadan/worker_compute.hpp"

namespace kadan::serving {
namespace {
constexpr Bytes gib = 1024ULL * 1024 * 1024;
// Media engines admit a device only with this much headroom (media_plan).
constexpr Bytes device_headroom = compute_context_bytes + 2 * gib;
// ChatEngine keeps this much host memory free beside its plan.
constexpr Bytes text_host_headroom = 4 * gib;
// Engine admission floors, from each engine's load().
constexpr Bytes image_host_floor = 16 * gib;
constexpr Bytes video_device_floor = 3 * gib;
constexpr Bytes decision_envelope = 3 * gib;
// Video prompts beyond this length no longer grow the text stage.
constexpr std::size_t max_prompt_bytes = 32000;

std::string default_model(Workload workload, const ModelStore& store) {
    switch (workload) {
    case Workload::llm:
        throw RequestError(
            "Language inference uses the LFM3 worker; native requests require an "
            "explicit checkpoint.",
            422
        );
    case Workload::decision:
        return "laya";
    case Workload::formatting:
        return "superwhisper-s1-mini";
    case Workload::speech:
        return "whisper-" + store.whisper_selected();
    case Workload::image:
        return "qwen-image-2.1-turbo";
    case Workload::tts:
        return "qwen-tts-1.7b-custom";
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
    const std::optional<Json>& requested_context
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
    case Workload::formatting:
        // S1-mini runs on the host with its weights resident as floats.
        return footprint(model, 2 * weights + gib, 2 * weights + gib, 0, 0, 0);
    case Workload::speech:
        return footprint(
            model,
            gib,
            weights + gib,
            device_headroom,
            weights + device_headroom,
            device_headroom
        );
    case Workload::tts:
        return footprint(
            model,
            2 * gib,
            weights + 2 * gib,
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
        break;
    }
    if (!store.complete(entry))
        // Before the download only the catalog size is known.
        return footprint(
            model,
            text_host_headroom,
            weights + text_host_headroom,
            device_headroom,
            weights + device_headroom,
            device_headroom
        );
    // The MoE engine pins non-expert weights and its context state in VRAM and
    // caches routed experts in whatever budget remains, so it runs on far less
    // than its full size and prefers all of it. Routed experts are the rank-3
    // tensors with more than one expert. Device context state mirrors the host
    // state that layout.host already counts.
    const auto root = store.root() / (model + "-" + entry.revision);
    // The same context chat admission uses: the saved limit, else the
    // checkpoint's supported one.
    const auto config = read_document(root / "config.json");
    // A queued load's explicit context wins, as in chat preparation.
    const auto saved = requested_context ? *requested_context : store.context(model);
    const auto context = saved.is_null() ? config.value("text_config", config)
                                               .at("max_position_embeddings")
                                               .get<std::size_t>()
                                         : saved.get<std::size_t>();
    const std::atomic_bool cancel = false;
    const auto layout = text::plan(root.c_str(), context, &cancel);
    const checkpoint::GgufCheckpoint gguf(root.string(), cancel);
    Bytes experts = 0;
    for (std::size_t i = 0; i < gguf.tensor_count(); ++i) {
        const auto& tensor = gguf.tensor_at(i);
        if (tensor.rank == 3 && tensor.shape[2] > 1)
            experts += tensor.bytes;
    }
    const Bytes pinned = layout.packed > experts ? layout.packed - experts : 0;
    return footprint(
        model,
        layout.host + text_host_headroom,
        layout.host + layout.packed + text_host_headroom,
        pinned + layout.host + device_headroom,
        layout.packed + layout.host + device_headroom,
        device_headroom
    );
}
}  // namespace kadan::serving
