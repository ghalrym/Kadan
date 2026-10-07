#include "kadan/checkpoint.hpp"

#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::invalid_argument("usage: kadan-checkpoint-inspect ROOT SHARD METADATA_BUDGET_BYTES");
        const std::string_view text(argv[3]); std::size_t bytes = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), bytes);
        if (error != std::errc{} || end != text.data() + text.size() || bytes == 0)
            throw std::invalid_argument("invalid_budget");
        auto budget = std::make_shared<kadan::checkpoint::MemoryBudget>(bytes);
        kadan::checkpoint::Shard shard(argv[1], argv[2], budget);
        std::cout << "tensors=" << shard.tensor_count() << " retained_metadata_bytes=" << budget->used() << '\n';
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 2;
    }
}
