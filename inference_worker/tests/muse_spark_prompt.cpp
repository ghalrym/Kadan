#include "kadan/muse_spark_prompt.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <tuple>
#include <utility>

using namespace kadan::serving;

namespace {
namespace m = muse_spark;
using Piece = std::pair<unsigned, std::string>;

Piece text(std::string value) {
    return {1000, std::move(value)};
}

Piece special(unsigned id) {
    // The parser must hide control tokens even if the decoder returns text.
    return {id, "<|control|>"};
}

std::pair<std::string, std::size_t> run(const std::vector<Piece>& tokens) {
    m::Stream stream;
    std::string output;
    std::size_t visible = 0;
    for (const auto& [id, decoded] : tokens) {
        output += stream.push(id, decoded);
        visible += stream.visible();
    }
    output += stream.finish();
    assert(!stream.visible());
    assert(stream.finish().empty());
    assert(stream.push(m::start, "").empty());
    return {output, visible};
}

void stream_units() {
    static_assert(m::is_stop(m::end_of_text) && m::is_stop(m::eot));
    static_assert(!m::is_stop(m::eom) && !m::is_stop(m::message));

    // Generation resumes inside the partial assistant header. Each ordinary
    // token may contain only part of the recipient; boundaries are irrelevant.
    auto [output, visible] = run(
        {text(" to="),
         text("se"),
         text("lf"),
         special(m::message),
         text("Think"),
         text(" first."),
         special(m::eom),
         special(m::start),
         text("assistant to=commentary"),
         special(m::message),
         text("Checking. "),
         special(m::eom),
         special(m::start),
         text("assistant to=user"),
         special(m::message),
         text("Four"),
         text("."),
         special(m::eot),
         text("stray"),
         special(m::start),
         text("assistant to=user"),
         special(m::message),
         text("after stop")}
    );
    assert(output == "<think>Think first.</think>Checking. Four." && visible == 3);

    // New message boundaries and length truncation both close reasoning.
    std::tie(output, visible) = run(
        {text(" to=self"),
         special(m::message),
         text("Reason"),
         special(m::start),
         text("assistant to=user"),
         special(m::message),
         text("Answer")}
    );
    assert(output == "<think>Reason</think>Answer" && visible == 1);
    std::tie(output, visible) =
        run({text(" to=self"), special(m::message), text("Unfinished")});
    assert(output == "<think>Unfinished</think>" && visible == 0);

    for (const auto stop : {m::eot, m::end_of_text}) {
        std::tie(output, visible) =
            run({text(" to=self"), special(m::message), text("Reason"), special(stop)});
        assert(output == "<think>Reason</think>" && visible == 0);
    }

    // Tool recipients and role impersonations cannot become visible answers.
    for (const auto header :
         {"assistant to=functions.lookup",
          "user to=user",
          "assistant to=user extra",
          "assistant to=unknown",
          "assistant",
          "assistant to=self extra"}) {
        std::tie(output, visible) = run(
            {special(m::start),
             text(header),
             special(m::message),
             text("hidden"),
             special(m::eom),
             text("also hidden"),
             special(m::start),
             text("assistant to=user"),
             special(m::message),
             text("Answer"),
             special(m::end_of_text)}
        );
        assert(output == "Answer" && visible == 1);
    }

    // Both a single large decoded token and split overflow fail closed.
    for (const auto& pieces : std::vector<std::vector<Piece>>{
             {text(std::string(1024, 'x')), special(m::message), text("hidden")},
             {text(" to=user"),
              text(std::string(512, ' ')),
              special(m::message),
              text("hidden")},
             {text(" to=se")}
         }) {
        std::tie(output, visible) = run(pieces);
        assert(output.empty() && visible == 0);
    }

    // Empty decoder fragments and structural tokens do not consume answer
    // budget or expose their spellings.
    std::tie(output, visible) = run(
        {text(" to=user"),
         special(m::message),
         text(""),
         special(m::begin_of_text),
         special(m::pad),
         special(m::message),
         text("Answer"),
         special(m::eot)}
    );
    assert(output == "Answer" && visible == 1);
}

bool rejects(const std::vector<ChatMessage>& messages, std::size_t limit = 16384) {
    try {
        muse_spark::prompt(messages, "2026-10-15", limit);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
}  // namespace

int main() {
    // Expected bytes were rendered independently by Jinja2 3.1.6 from the
    // complete public Hub template, not constructed from the native renderer.
    // The generation extension only unwraps its training-only generation block.
    const auto path = std::filesystem::path(__FILE__).parent_path() / "fixtures"
                      / "muse_spark_prompts.json";
    std::ifstream file(path);
    assert(file);
    const auto fixture = Json::parse(file);
    assert(fixture.at("revision") == "0fb02478b02565b2ce7fc57e6051c91172774ac9");
    assert(
        fixture.at("template_sha256")
        == "2341a2d3b1f1b0ac92b6237d46e2fff495109bd3305cacdea54726f5e98411db"
    );
    for (const auto& example : fixture.at("cases")) {
        std::vector<ChatMessage> messages;
        for (const auto& message : example.at("messages"))
            messages.push_back({message.at("role"), message.at("content")});
        const auto expected = example.at("expected").get<std::string>();
        const auto date = example.at("date").get<std::string>();
        const auto rendered = muse_spark::prompt(messages, date, 16384);
        if (rendered != expected) {
            std::cerr << "Template mismatch: " << example.at("name") << '\n';
            return 1;
        }
        // Admission includes expansion, the system preamble and final prefix.
        assert(muse_spark::prompt(messages, date, expected.size()) == expected);
        bool exceeded = false;
        try {
            muse_spark::prompt(messages, date, expected.size() - 1);
        } catch (const std::invalid_argument&) {
            exceeded = true;
        }
        assert(exceeded);
    }
    assert(rejects({}));
    assert(rejects({{"user", "Hi"}}, 0));
    assert(rejects({{"user", "Hi"}}, 1));
    assert(rejects({{"system", "Only instructions"}}));
    assert(rejects({{"assistant", "Only answer"}}));
    assert(rejects({{"developer", "Instructions"}, {"user", "Hi"}}));
    assert(rejects({{"tool", "{}"}, {"user", "Hi"}}));
    assert(rejects({{"user", std::string(16385, 'x')}}));
    assert(rejects({{"user", std::string(10000, '<')}}, 1000));
    stream_units();
    std::cout << "Muse Spark text prompt and recipient stream units passed\n";
}
