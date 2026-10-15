#include "kadan/muse_spark_prompt.hpp"

#include <cassert>
#include <fstream>
#include <iostream>

using namespace kadan::serving;

namespace {
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
    std::cout << "Muse Spark text prompt units passed\n";
}
