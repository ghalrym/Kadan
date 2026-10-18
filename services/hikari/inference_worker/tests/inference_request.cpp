// Request parsing only: no model acquisition, preparation or inference.
#include "kadan/inference_request.hpp"

#include <cassert>
#include <iostream>

using namespace kadan::serving;

int main() {
    const Json envelope{
        {"id", std::string(32, 'a')},
        {"command", "submit"},
        {"operation", "generate"},
        {"input", {{"feature", "image"}, {"prompt", "fox"}, {"count", 1}}}
    };
    const auto missing = parse_request(envelope);
    assert(!std::get<ImageRequest>(missing.input).batch_collection_target);
    for (const auto& target : {Json(nullptr), Json(1), Json(3), Json(32)}) {
        auto value = envelope;
        value["input"]["batch_collection_target"] = target;
        const auto parsed = parse_request(value);
        const auto& input = std::get<ImageRequest>(parsed.input);
        if (target.is_null())
            assert(!input.batch_collection_target);
        else
            assert(input.batch_collection_target == target.get<unsigned>());
        assert(input.count == 1);
    }
    for (const auto& invalid :
         {Json(0),
          Json(-1),
          Json(33),
          Json(1.5),
          Json(true),
          Json("2"),
          Json(4294967297ULL)}) {
        auto value = envelope;
        value["input"]["batch_collection_target"] = invalid;
        bool rejected = false;
        try {
            (void)parse_request(value);
        } catch (const RequestError& error) {
            rejected = error.status == 422;
        }
        assert(rejected);
    }
    auto reference = envelope;
    reference["input"].update(
        {{"image", "data:image/png;base64,cG5n"},
         {"mode", "reference"},
         {"batch_collection_target", 2}}
    );
    const auto parsed = parse_request(reference);
    assert(std::get<ImageRequest>(parsed.input).reference);
    assert(std::get<ImageRequest>(parsed.input).batch_collection_target == 2);
    auto speech = envelope;
    speech["input"] = {
        {"feature", "tts"},
        {"script", "Hello"},
        {"model_id", "drama-3"},
        {"voice", {{"speaker", "default"}}},
        {"stream", true},
        {"expected_tier", "balanced"},
        {"require_installed", true}
    };
    const auto speech_request = parse_request(speech);
    const auto& audio = std::get<SpeechRequest>(speech_request.input);
    assert(audio.stream && audio.require_installed);
    assert(audio.expected_tier == "balanced" && audio.model == "drama-3");
    std::cout << "Inference request parser units passed\n";
}
