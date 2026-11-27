// Request parsing only: no model acquisition, preparation or inference.
#include "kadan/inference_request.hpp"

#include <cassert>
#include <iostream>

#include "kadan/chat_prompt.hpp"
#include "kadan/inkling_gpu.hpp"
#include "kadan/inkling_variant.hpp"

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
         {"reference_images", Json::array({"data:image/png;base64,cG5n"})},
         {"mode", "reference"},
         {"batch_collection_target", 2}}
    );
    const auto parsed = parse_request(reference);
    assert(std::get<ImageRequest>(parsed.input).reference);
    assert(std::get<ImageRequest>(parsed.input).reference_images.size() == 1);
    assert(std::get<ImageRequest>(parsed.input).batch_collection_target == 2);
    for (const auto& bad_refs : {Json::array(), Json::array({"a", "b", "c", "d"}), Json("not-array")}) {
        auto bad = reference;
        bad["input"]["reference_images"] = bad_refs;
        bool rejected = false;
        try {
            (void)parse_request(bad);
        } catch (const RequestError& error) {
            rejected = error.status == 422;
        }
        assert(rejected);
    }
    {
        auto bad_mode = envelope;
        bad_mode["input"].update(
            {{"image", "data:image/png;base64,cG5n"},
             {"reference_images", Json::array({"data:image/png;base64,cG5n"})},
             {"mode", "source"}}
        );
        bool rejected = false;
        try {
            (void)parse_request(bad_mode);
        } catch (const RequestError& error) {
            rejected = error.status == 422;
        }
        assert(rejected);
    }
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
    chat["input"] = {
        {"feature", "llm"},
        {"max_tokens", 2048},
        {"reasoning_tokens", 128},
        {"reasoning_effort", "medium"},
        {"tools",
         Json::array(
             {{{"type", "function"},
               {"function",
                {{"name", "bash"},
                 {"description", "Run shell"},
                 {"parameters", {{"type", "object"}}}}}}}
         )},
        {"messages",
         Json::array(
             {{{"role", "user"},
               {"content",
                Json::array(
                    {{{"type", "text"}, {"text", "Look at this"}},
                     {{"type", "image_url"},
                      {"image_url", {{"url", "data:image/png;base64,iVBORw0KGgo="}}}},
                     {{"type", "text"}, {"text", " then that"}}}
                )}},
              {{"role", "assistant"},
               {"content", ""},
               {"reasoning_content", "checking"},
               {"tool_calls",
                Json::array(
                    {{{"id", "call_1"},
                      {"type", "function"},
                      {"function",
                       {{"name", "bash"}, {"arguments", "{\"cmd\":\"ls\"}"}}}}}
                )}},
              {{"role", "tool"},
               {"tool_call_id", "call_1"},
               {"name", "bash"},
               {"content", "ok"}}}
         )}
    };
    {
        const auto parsed_chat = parse_request(chat);
        const auto& req = std::get<ChatRequest>(parsed_chat.input);
        assert(req.max_output_tokens == 2048);
        assert(req.reasoning_tokens == 128);
        assert(req.reasoning_effort == "medium");
        assert(req.tools.size() == 1 && req.tools[0].name == "bash");
        assert(req.messages.size() == 3);
        assert(req.messages[0].text == "Look at this then that");
        assert(req.messages[0].image_offsets == std::vector<std::size_t>{12});
        const auto prompt = chat_prompt(req.messages, 4096).text;
        assert(prompt.find("Look at this") < prompt.find("<|content_image|>"));
        assert(prompt.find("<|content_image|>") < prompt.find(" then that"));
        assert(req.messages[0].images.size() == 1);
        assert(req.messages[1].reasoning_content == "checking");
        assert(req.messages[1].tool_calls.size() == 1);
        assert(req.messages[1].tool_calls[0].id == "call_1");
        assert(req.messages[2].tool_call_id == "call_1");
        assert(req.messages[2].name == "bash");
        assert(parsed_chat.priority == Priority::interactive);
    }
    {
        auto bg = chat;
        bg["input"]["priority"] = "background";
        assert(parse_request(bg).priority == Priority::background);
        bg["input"]["priority"] = "interactive";
        assert(parse_request(bg).priority == Priority::interactive);
        bg["input"]["priority"] = "urgent";
        bool bad_priority = false;
        try {
            (void)parse_request(bg);
        } catch (const RequestError& error) {
            bad_priority = error.status == 422;
        }
        assert(bad_priority);
    }
    {
        auto sampled = chat;
        sampled["input"]["temperature"] = 0.7;
        sampled["input"]["top_p"] = 0.95;
        sampled["input"]["top_k"] = 40;
        sampled["input"]["min_p"] = 0.05;
        sampled["input"]["repetition_penalty"] = 1.08;
        sampled["input"]["presence_penalty"] = 0.4;
        sampled["input"]["seed"] = 12345;
        const auto& req = std::get<ChatRequest>(parse_request(sampled).input);
        assert(req.temperature && *req.temperature > 0.69f && *req.temperature < 0.71f);
        assert(req.top_p && *req.top_p > 0.94f && *req.top_p < 0.96f);
        assert(req.top_k == 40u);
        assert(req.min_p && *req.min_p > 0.049f && *req.min_p < 0.051f);
        assert(
            req.repetition_penalty && *req.repetition_penalty > 1.07f
            && *req.repetition_penalty < 1.09f
        );
        assert(
            req.presence_penalty && *req.presence_penalty > 0.39f
            && *req.presence_penalty < 0.41f
        );
        assert(req.seed == 12345u);

        for (const auto& [field, invalid] : std::vector<std::pair<std::string, Json>>{
                 {"temperature", Json(-0.1)},
                 {"temperature", Json(2.1)},
                 {"temperature", Json(true)},
                 {"top_p", Json(0.0)},
                 {"top_p", Json(1.1)},
                 {"top_k", Json(-1)},
                 {"top_k", Json(1.5)},
                 {"min_p", Json(-0.01)},
                 {"min_p", Json(1.01)},
                 {"repetition_penalty", Json(0.9)},
                 {"repetition_penalty", Json(2.1)},
                 {"presence_penalty", Json(-2.1)},
                 {"presence_penalty", Json(2.1)},
                 {"seed", Json(-1)},
                 {"seed", Json(false)}
             }) {
            auto bad = chat;
            bad["input"][field] = invalid;
            bool rejected = false;
            try {
                (void)parse_request(bad);
            } catch (const RequestError& error) {
                rejected = error.status == 422;
            }
            assert(rejected);
        }
    }
    auto decisions = envelope;
    decisions["input"] = {
        {"feature", "decisions"}, {"state", "text"}, {"questions", Json::array()}
    };
    bool empty_decisions_rejected = false;
    try {
        (void)parse_request(decisions);
    } catch (const RequestError& error) {
        empty_decisions_rejected = error.status == 422;
    }
    assert(empty_decisions_rejected);

    decisions["input"]["questions"] = Json::array({
        {{"key", "urgent"},
         {"type", "Noul"},
         {"instructions", "Is this urgent?"},
         {"trueWhen", "Needs action now"},
         {"falseWhen", "Can wait"}}
    });
    const auto parsed_decision = parse_request(decisions);
    assert(parsed_decision.workload == kadan::Workload::llm);
    assert(!std::get<DecisionRequest>(parsed_decision.input).model.has_value());
    assert(std::get<DecisionRequest>(parsed_decision.input).questions.size() == 1);

    auto audio_call = envelope;
    audio_call["input"] = {
        {"feature", "audio_call"},
        {"mode", "tts"},
        {"script", "Hello world"}
    };
    const auto parsed_a = parse_request(audio_call);
    assert(parsed_a.workload == kadan::Workload::lfm3_audio);

    auto llm_sel = envelope;
    llm_sel["operation"] = "configure";
    llm_sel["input"] = {
        {"feature", "llm_selection"},
        {"model_id", "inkling-2-mini-abliterated-q8-0"}
    };
    const auto parsed_sel = parse_request(llm_sel);
    assert(std::get<LlmSelectionRequest>(parsed_sel.input).model == "inkling-2-mini-abliterated-q8-0");

    // Weight variant parsing for llm and decisions + Inkling variant/tier helpers.
    {
        auto vchat = chat;
        vchat["input"]["weight_variant"] = "base";
        assert(
            std::get<ChatRequest>(parse_request(vchat).input).weight_variant == "base"
        );
        vchat["input"]["weight_variant"] = "abliterated";
        assert(
            std::get<ChatRequest>(parse_request(vchat).input).weight_variant
            == "abliterated"
        );
        vchat["input"]["weight_variant"] = "uncensored";
        bool bad_chat_variant = false;
        try {
            (void)parse_request(vchat);
        } catch (const RequestError& error) {
            bad_chat_variant = error.status == 422;
        }
        assert(bad_chat_variant);

        auto vdec = decisions;
        vdec["input"]["weight_variant"] = "base";
        assert(
            std::get<DecisionRequest>(parse_request(vdec).input).weight_variant
            == "base"
        );
        vdec["input"]["weight_variant"] = "invalid";
        bool bad_dec_variant = false;
        try {
            (void)parse_request(vdec);
        } catch (const RequestError& error) {
            bad_dec_variant = error.status == 422;
        }
        assert(bad_dec_variant);

        using kadan::InklingSwapTier;
        using kadan::InklingVariant;
        assert(
            kadan::resolve_inkling_variant_model(
                "inkling-2-mini-abliterated-iq4-nl", InklingVariant::base
            )
            == "inkling-2-mini-iq4-nl"
        );
        assert(
            kadan::resolve_inkling_variant_model(
                "inkling-2-mini-iq4-nl", InklingVariant::abliterated
            )
            == "inkling-2-mini-abliterated-iq4-nl"
        );
        assert(
            kadan::resolve_inkling_variant_model(
                "inkling-2-mini-abliterated-q8-0", InklingVariant::base
            )
            == "inkling-2-mini-q8-0"
        );
        assert(
            kadan::resolve_inkling_variant_model(
                "inkling-2-mini-q8-0", InklingVariant::abliterated
            )
            == "inkling-2-mini-abliterated-q8-0"
        );
        assert(
            kadan::inkling_hot_swap_compatible(
                "inkling-2-mini-iq4-nl", "inkling-2-mini-abliterated-iq4-nl"
            )
        );
        assert(
            kadan::inkling_hot_swap_compatible(
                "inkling-2-mini-q8-0", "inkling-2-mini-abliterated-q8-0"
            )
        );
        assert(!kadan::inkling_hot_swap_compatible(
            "inkling-2-mini-iq4-nl", "inkling-2-mini-q8-0"
        ));
        assert(!kadan::inkling_hot_swap_compatible(
            "inkling-2-mini-iq4-nl", "inkling-2-mini-iq4-nl"
        ));

        // Exact 34 differing tensors: blk.8..23.attn_output.weight (16) and
        // blk.6..23.ffn_down_exps.weight (18).
        std::size_t attn_count = 0, down_count = 0;
        for (std::size_t i = 0; i < 24; ++i) {
            const std::string ao = "blk." + std::to_string(i) + ".attn_output.weight";
            const std::string fd = "blk." + std::to_string(i) + ".ffn_down_exps.weight";
            const std::string gu =
                "blk." + std::to_string(i) + ".ffn_gate_up_exps.weight";
            assert(kadan::is_inkling_variant_tensor(ao) == (i >= 8));
            assert(kadan::is_inkling_variant_tensor(fd) == (i >= 6));
            assert(!kadan::is_inkling_variant_tensor(gu));
            attn_count += kadan::is_inkling_variant_tensor(ao);
            down_count += kadan::is_inkling_variant_tensor(fd);
        }
        assert(attn_count == 16 && down_count == 18);
        assert(!kadan::is_inkling_variant_tensor("token_embd.weight"));
        assert(!kadan::is_inkling_variant_tensor("output.weight"));
        assert(!kadan::is_inkling_variant_tensor("blk.24.attn_output.weight"));

        constexpr std::size_t delta = 4550ULL << 20;  // ~4.24 GiB
        assert(
            kadan::select_inkling_swap_tier(delta, 10ULL << 30, 32ULL << 30)
            == InklingSwapTier::gpu_dual_buffer
        );
        assert(
            kadan::select_inkling_swap_tier(delta, 2ULL << 30, 16ULL << 30)
            == InklingSwapTier::pinned_host_ram
        );
        assert(
            kadan::select_inkling_swap_tier(delta, 1ULL << 30, 4ULL << 30)
            == InklingSwapTier::stream_repack
        );
    }

    std::cout << "Inference request parser units passed\n";
}
