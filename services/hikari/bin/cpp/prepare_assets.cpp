#include <iostream>

#include "kadan/preparation.hpp"

int main(int argc, char** argv) {
    using namespace kadan::serving;
    try {
        const std::atomic_bool cancel{false};
        preparation_check(
            argc >= 2, "Expected speech-tokenizer or tts-tokenizer"
        );
        const std::string command = argv[1];
        if (command == "speech-tokenizer" || command == "tts-tokenizer") {
            preparation_check(argc == 4, "Expected checkpoint and destination");
            export_speech_tokenizer(argv[2], argv[3], cancel);
        } else
            throw RequestError("Unknown asset conversion command");
        std::cout << "Assets prepared; no model execution.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
