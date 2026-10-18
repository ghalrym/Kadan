// Pure prompt formatting and validation; no model loading or inference.
#include "kadan/drama_prompt.hpp"

#include <cassert>
#include <iostream>

int main() {
    using kadan::drama::prompt_segments;
    const auto parts = prompt_segments("Hello.");
    assert(parts.size() == 7);
    assert(parts[0] == "<|im_start|>system\n");
    assert(parts[1] == "convert the provided text to speech");
    assert(parts[4] == "<|speaker:0|> Hello.");
    assert(parts[6] == "<|im_start|>assistant\n<|voice|>");
    assert(prompt_segments("  <|speaker:0|> Hello.")[4] == "  <|speaker:0|> Hello.");
    assert(
        prompt_segments("\xe3\x80\x80<|speaker:0|> Hello.")[4]
        == "\xe3\x80\x80<|speaker:0|> Hello."
    );
    for (const auto text :
         {"",
          " \n\xe3\x80\x80",
          "<|speaker:1|> hello",
          "<|speaker:00|> hi",
          "<|speaker:0> hi",
          "<|semantic:123|>",
          "<|im_end|>",
          "<|voice|>"}) {
        bool rejected = false;
        try {
            (void)prompt_segments(text);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }
    std::cout << "Drama prompt units passed\n";
}
