#pragma once
#include <array>
#include <atomic>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
namespace kadan::decision {
using Json = nlohmann::ordered_json;
Json parse_json(std::string_view text);
Json read_json(const std::string &path, std::size_t limit);
void check(bool, const char *);
void cancel_check(const std::atomic_bool &);
class Tokenizer {
  public:
    explicit Tokenizer(const Json &);
    ~Tokenizer();
    Tokenizer(const Tokenizer &) = delete;
    std::vector<int> encode(const std::string &, const std::atomic_bool &) const;
    int cls = 0, sep = 0, mask = 0;

  private:
    std::array<std::string, 256> bytes_;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, std::pair<int, int>> merges_;
    struct Added {
        std::string text;
        int id;
        bool normalized;
    };
    std::vector<Added> added_;
    void *pattern_ = nullptr;
    void ordinary(std::string_view, std::vector<int> &, const std::atomic_bool &) const;
};
} // namespace kadan::decision
