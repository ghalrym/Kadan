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
    auto chat = envelope;
    chat["operation"] = "completion";
    chat["input"] = {
        {"feature", "llm"}, {"conversation_id", "chat-1"}, {"messages", Json::array()}
    };
    assert(std::get<ChatRequest>(parse_request(chat).input).reuse_prefix);
    chat["input"]["reuse_prefix"] = false;
    assert(!std::get<ChatRequest>(parse_request(chat).input).reuse_prefix);
    std::string unicode_id;
    for (int i = 0; i < 65; ++i)
        unicode_id += "\xc3\xa9";
    chat["input"]["conversation_id"] = unicode_id;
    assert(
        std::get<ChatRequest>(parse_request(chat).input).conversation_id == unicode_id
    );
    std::cout << "Inference request parser units passed\n";
}
