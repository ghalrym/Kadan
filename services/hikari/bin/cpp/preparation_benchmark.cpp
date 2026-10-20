#include <chrono>
#include <ctime>
#include <iostream>

#include "kadan/checkpoint.hpp"
#include "kadan/gguf_checkpoint.hpp"
#include "kadan/gguf_text.hpp"
#include "kadan/preparation.hpp"

// Metadata only: no engine, tensor reads, model execution or GPU context.
int main(int argc, char** argv) {
    using namespace kadan::serving;
    try {
        preparation_check(
            argc == 4, "Expected model store, model id, and sample count"
        );
        const auto count = std::stoul(argv[3]);
        preparation_check(count >= 1 && count <= 100, "Expected 1 to 100 samples");
        const std::atomic_bool cancel{false};
        ModelStore store(
            argv[1],
            worker_data_directory(),
            [](const auto&, const auto&, const auto&, bool, auto) -> std::int64_t {
                throw std::runtime_error("Benchmark downloads are forbidden");
            }
        );
        const auto& entry = store.entry(argv[2]);
        const auto root = store.root() / (entry.id + "-" + entry.revision);
        preparation_check(
            entry.layout == "gguf" && store.complete(entry)
                && fs::is_regular_file(root / "config.json")
                && fs::is_regular_file(root / "tokenizer.json"),
            "Benchmark requires an existing complete GGUF checkpoint and derived assets"
        );
        const auto sample = [&](const char* phase, std::size_t i, const auto& work) {
            const auto start = std::chrono::steady_clock::now();
            const auto cpu = std::clock();
            work();
            std::cout << Json{
                {"phase", phase},
                {"sample", i},
                {"wall_ms",
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - start
                 )
                     .count()},
                {"cpu_ms", 1000.0 * (std::clock() - cpu) / CLOCKS_PER_SEC}
            }.dump() << '\n';
        };
        // Substeps are independent probes, not additive spans of acquire.
        // Sample zero includes the store's first acquisition; retain it separately.
        for (std::size_t i = 0; i <= count; ++i) {
            sample("complete", i, [&] {
                preparation_check(store.complete(entry), "Checkpoint is incomplete");
            });
            sample("metadata_parse_validate", i, [&] {
                kadan::checkpoint::GgufMetadata metadata(
                    (root / entry.asset_name).string(), cancel
                );
                (void)kadan::text::validate_metadata(metadata);
            });
            sample("prepare_gguf_assets", i, [&] {
                prepare_gguf_assets(root, entry.asset_name, cancel);
            });
            sample("validate_config", i, [&] {
                kadan::text::validate_config(
                    root.c_str(),
                    std::make_shared<kadan::checkpoint::MemoryBudget>(4ULL << 20)
                );
            });
            sample("acquire", i, [&] {
                preparation_check(
                    store.acquire(entry.id, cancel) == root,
                    "Unexpected checkpoint path"
                );
            });
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
