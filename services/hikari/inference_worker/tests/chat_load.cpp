// Control-only load completion on an unloaded engine; no load, prepare or inference.
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "kadan/inference_engines.hpp"
#include "kadan/preparation.hpp"

using namespace kadan::serving;

namespace {
struct Directory {
    fs::path path;

    Directory() {
        std::string pattern = "/tmp/kadan-chat-load-unit-XXXXXX";
        const auto* value = mkdtemp(pattern.data());
        assert(value);
        path = value;
    }

    ~Directory() {
        fs::remove_all(path);
    }
};
}  // namespace

int main() {
    Directory temporary;
    const auto root = temporary.path / "settings", data = temporary.path / "data";
    fs::create_directories(root);
    fs::create_directories(data);
    write_document(
        data / "catalog.json",
        {{"assets", Json::array()},
         {"models",
          Json::array(
              {{{"id", "chat"},
                {"repo_id", "unit/model"},
                {"revision", "revision"},
                {"kind", "llm"},
                {"layout", "composite"},
                {"inference_available", true},
                {"required_files", Json::array()},
                {"manifest", Json::array()}}}
          )}}
    );
    auto engine = chat_engine(
        nullptr,
        [](const char*, std::size_t) {
            throw std::runtime_error("Unexpected progress");
        },
        [](const std::string&) { throw std::runtime_error("Unexpected content"); }
    );
    const std::atomic_bool running{false};
    bool preserved = true;
    for (const auto saved : {4096, 16384}) {
        const Json defaults = {{"chat", saved}, {"other-model", 32768}};
        write_document(root / "context.json", defaults);
        const auto original = read_bytes(root / "context.json", 4096);
        for (const auto& override : std::vector<std::optional<Json>>{
                 std::nullopt, Json(8192), std::nullopt, Json(nullptr), std::nullopt
             }) {
            Json body = {{"feature", "llm"}, {"model_id", "chat"}};
            if (override)
                body["context_limit"] = *override;
            auto request = parse_request(
                {{"id", std::string(32, 'a')},
                 {"command", "submit"},
                 {"operation", "load"},
                 {"input", body}}
            );
            request.model = "chat";
            assert(engine->execute(request, running) == Json::object());
            ModelStore reopened(root, data);
            const bool unchanged =
                reopened.context("chat") == saved
                && read_bytes(root / "context.json", 4096) == original;
            if (!unchanged)
                std::cerr << "Load completion changed saved context " << saved
                          << " for " << body << ": "
                          << read_document(root / "context.json") << '\n';
            preserved &= unchanged;
            write_document(root / "context.json", defaults);
        }
    }
    assert(preserved);
    std::cout << "Chat load-completion units passed; no model loaded or executed.\n";
}
