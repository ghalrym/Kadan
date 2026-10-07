#pragma once
#include "kadan/checkpoint.hpp"
#include <string>

namespace kadan::checkpoint::json {
// Original bounded ASCII JSON DOM for model metadata, never tensor payloads.
// Every variable-size allocation uses the caller's admitted MemoryBudget.
struct Value {
    enum class Kind { object, array, string, number, boolean, null } kind;
    std::pmr::string name, text;
    std::pmr::vector<Value> children;
    Value(Kind type, std::pmr::memory_resource* r) : kind(type), name(r), text(r), children(r) {}
    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;
    Value(Value&&) = default;
    Value& operator=(Value&&) = default;
    const Value& at(std::string_view key) const;
    const Value* find(std::string_view key) const;
    std::string_view string() const;
    std::uint64_t integer() const;
    double number() const;
    bool boolean() const;
};
Value read(const char* root, std::string_view basename, std::size_t byte_limit,
           const std::shared_ptr<MemoryBudget>& budget);
} // namespace kadan::checkpoint::json
