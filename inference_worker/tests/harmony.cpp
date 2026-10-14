#include "kadan/harmony.hpp"

#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

using namespace kadan::serving;
namespace h = kadan::serving::harmony;

namespace {
using Piece = std::pair<unsigned, std::string>;

// Text tokens use an ordinary id; special tokens decode to nothing.
Piece text(std::string value) {
    return {1000, std::move(value)};
}

Piece special(unsigned id) {
    return {id, ""};
}

std::pair<std::string, std::size_t> run(const std::vector<Piece>& tokens) {
    h::Stream stream;
    std::string out;
    std::size_t visible = 0;
    for (const auto& [id, value] : tokens) {
        out += stream.push(id, value);
        visible += stream.visible();
    }
    return {out + stream.finish(), visible};
}

bool rejects(const std::vector<ChatMessage>& messages, std::size_t limit = 4096) {
    try {
        h::prompt(messages, "2026-10-13", limit);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
}  // namespace

int main() {
    assert(h::is_stop(h::return_) && h::is_stop(h::call) && h::is_stop(h::end_of_text));
    assert(!h::is_stop(h::end) && !h::is_stop(h::message));
    const std::string system =
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by "
        "OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-10-13\n\nReasoning: "
        "low\n\n# Valid channels: analysis, commentary, final. Channel must be "
        "included for every message.<|end|>";
    assert(
        h::prompt({{"user", "Hi"}}, "2026-10-13", 4096)
        == system + "<|start|>user<|message|>Hi<|end|><|start|>assistant"
    );
    // System instructions become developer; earlier reasoning is dropped.
    assert(
        h::prompt(
            {{"system", "Be brief."},
             {"user", "2+2?"},
             {"assistant", "<think>Add them.</think>\n4 "},
             {"user", "Thanks"}},
            "2026-10-13",
            4096
        )
        == system
               + "<|start|>developer<|message|># Instructions\n\nBe brief.<|end|>"
                 "<|start|>user<|message|>2+2?<|end|>"
                 "<|start|>assistant<|channel|>final<|message|>4<|end|>"
                 "<|start|>user<|message|>Thanks<|end|><|start|>assistant"
    );
    assert(rejects({}));
    assert(rejects({{"system", "Only instructions"}}));
    assert(rejects({{"tool", "{}"}}));
    assert(rejects({{"user", std::string(5000, 'x')}}));

    // Reasoning, then the answer: analysis becomes a think block.
    auto [answer, visible] = run(
        {special(h::channel),
         text("analysis"),
         special(h::message),
         text("Simple"),
         text(" sum."),
         special(h::end),
         special(h::start),
         text("assistant"),
         special(h::channel),
         text("final"),
         special(h::message),
         text("Four"),
         text("."),
         special(h::return_)}
    );
    assert(answer == "<think>Simple sum.</think>Four." && visible == 2);

    // A direct final answer, including a content-type constraint.
    std::tie(answer, visible) = run(
        {special(h::channel),
         text("final"),
         special(h::constrain),
         text("json"),
         special(h::message),
         text("{}"),
         special(h::return_)}
    );
    assert(answer == "{}" && visible == 1);

    // Length limits close open reasoning; tool calls are never shown.
    std::tie(answer, visible) =
        run({special(h::channel), text("analysis"), special(h::message), text("Hmm")});
    assert(answer == "<think>Hmm</think>" && visible == 0);
    std::tie(answer, visible) = run(
        {text(" to=functions.lookup"),
         special(h::channel),
         text("commentary json"),
         special(h::message),
         text("{\"q\":1}"),
         special(h::call)}
    );
    assert(answer.empty() && visible == 0);

    // Preamble commentary is user-visible; tokens after a stop are ignored.
    std::tie(answer, visible) = run(
        {special(h::channel),
         text("commentary"),
         special(h::message),
         text("Checking."),
         special(h::end),
         text("stray")}
    );
    assert(answer == "Checking." && visible == 1);
    std::cout << "Harmony prompt and channel parsing checks passed; no inference\n";
}
