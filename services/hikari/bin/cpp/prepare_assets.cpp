#include <iostream>

#include "kadan/preparation.hpp"

int main(int argc, char** argv) {
    using namespace kadan::serving;
    try {
        const std::atomic_bool cancel{false};
        preparation_check(argc >= 2, "Expected whisper or whisper-assets");
        const std::string command = argv[1];
        if (command == "whisper") {
            preparation_check(
                argc == 5, "Expected checkpoint, SHA-256, and destination"
            );
            export_whisper(argv[2], argv[3], argv[4], cancel);
        } else if (command == "whisper-assets") {
            preparation_check(argc == 4, "Expected vocabulary and destination");
            export_whisper_assets(
                std::stoul(argv[2]), worker_data_directory() / "assets", argv[3], cancel
            );
        } else
            throw RequestError("Unknown asset conversion command");
        std::cout << "Assets prepared; no model execution.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
