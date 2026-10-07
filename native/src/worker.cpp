#include "kadan/resources.hpp"

#include <charconv>
#include <iostream>
#include <sstream>
#include <string_view>

namespace {
std::uint64_t number(std::string_view text) {
    std::uint64_t value = 0;
    auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument("invalid_integer");
    return value;
}
kadan::Workload workload(const std::string& name) {
    if (name == "llm") return kadan::Workload::llm;
    if (name == "image") return kadan::Workload::image;
    if (name == "video") return kadan::Workload::video;
    if (name == "speech") return kadan::Workload::speech;
    if (name == "tts") return kadan::Workload::tts;
    if (name == "decision") return kadan::Workload::decision;
    throw std::invalid_argument("invalid_workload");
}
// Drain an oversized frame without allocating proportional to untrusted input.
bool frame(std::string& line, bool& oversized) {
    line.clear(); oversized = false;
    char ch;
    bool any = false;
    while (std::cin.get(ch)) {
        any = true;
        if (ch == '\n') return true;
        if (line.size() < 4096) line += ch;
        else oversized = true;
    }
    return any;
}
std::string dispatch(kadan::Resources& resources, const std::vector<std::string>& args, std::size_t budgets) {
    if (args.empty()) throw std::invalid_argument("invalid_command");
    const auto& command = args[0];
    if (command == "hello" && args.size() == 1) return "ok kadan-worker 1 accounting-only";
    if (command == "snapshot" && args.size() == 1) {
        const auto snapshot = resources.snapshot();
        std::ostringstream result;
        result << "ok " << snapshot.residents;
        for (auto bytes : snapshot.used) result << ' ' << bytes;
        return result.str();
    }
    if (command == "reserve" && args.size() == budgets + 2) {
        kadan::Footprint bytes;
        for (std::size_t i = 2; i < args.size(); ++i) bytes.push_back(number(args[i]));
        return "ok " + std::to_string(resources.reserve(workload(args[1]), std::move(bytes)));
    }
    if (args.size() != 2) throw std::invalid_argument("invalid_command");
    const auto handle = number(args[1]);
    if (command == "loaded") resources.loaded(handle);
    else if (command == "pin") resources.pin(handle);
    else if (command == "unpin") resources.unpin(handle);
    else if (command == "evict") resources.begin_eviction(handle);
    else if (command == "eviction_failed") resources.eviction_failed(handle);
    else if (command == "released") resources.released(handle);
    else throw std::invalid_argument("invalid_command");
    return "ok";
}
} // namespace

int main(int argc, char** argv) {
    try {
        // Explicit budgets only: no device probing, driver initialization or model loading.
        if (argc < 2 || argc > 66) throw std::invalid_argument("usage: kadan-worker HOST_BYTES [GPU_BYTES ...] (max 64 GPUs)");
        kadan::Footprint capacity;
        for (int i = 1; i < argc; ++i) capacity.push_back(number(argv[i]));
        kadan::Resources resources(capacity);
        std::string line;
        bool oversized;
        while (frame(line, oversized)) {
            try {
                if (oversized) throw std::invalid_argument("frame_too_large");
                std::istringstream input(line);
                std::vector<std::string> args;
                for (std::string arg; input >> arg;) args.push_back(std::move(arg));
                const auto reply = dispatch(resources, args, capacity.size());
                std::cout << reply << '\n' << std::flush;
            } catch (const std::exception& error) {
                std::cout << "error " << error.what() << '\n' << std::flush;
            }
            if (!std::cout) return 1;
        }
        return std::cin.bad() ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
