#include <array>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "kadan/device_weights.hpp"

#ifndef KADAN_TRACE_WEIGHT_CACHE
#error "device_weight_trace requires KADAN_TRACE_WEIGHT_CACHE"
#endif

namespace {
using namespace kadan;
using Json = nlohmann::json;

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

class StderrCapture {
public:
    explicit StderrCapture(bool discard = false)
        : file_(discard ? std::fopen("/dev/null", "w") : std::tmpfile()) {
        check(file_ != nullptr, "trace_capture_open");
        std::fflush(stderr);
        saved_ = ::dup(STDERR_FILENO);
        if (saved_ < 0 || ::dup2(::fileno(file_), STDERR_FILENO) < 0) {
            if (saved_ >= 0)
                ::close(saved_);
            std::fclose(file_);
            throw std::runtime_error("trace_capture_redirect");
        }
    }

    StderrCapture(const StderrCapture&) = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;

    ~StderrCapture() {
        restore();
        std::fclose(file_);
    }

    std::vector<Json> records() {
        restore();
        check(std::fseek(file_, 0, SEEK_SET) == 0, "trace_capture_rewind");
        std::array<char, 4096> line{};
        std::vector<Json> result;
        while (std::fgets(line.data(), int(line.size()), file_)) {
            check(std::strchr(line.data(), '\n') != nullptr, "trace_record_size");
            check(result.size() < 128, "trace_test_capture_bound");
            result.push_back(Json::parse(line.data()));
        }
        check(!std::ferror(file_), "trace_capture_read");
        return result;
    }

private:
    void restore() noexcept {
        if (saved_ >= 0) {
            std::fflush(stderr);
            if (::dup2(saved_, STDERR_FILENO) < 0)
                std::abort();
            ::close(saved_);
            saved_ = -1;
        }
    }

    FILE* file_;
    int saved_ = -1;
};

struct Fake final : WeightDeviceOperations {
    Bytes free_bytes = 1024;
    bool oom = false;
    std::map<void*, Bytes> allocations;
    unsigned copies = 0, frees = 0;
    std::uintptr_t next_pointer = 0;

    Bytes available_weight_bytes() override {
        return free_bytes;
    }

    void* allocate_weight(Bytes size) override {
        if (oom)
            return nullptr;
        auto* pointer = reinterpret_cast<void*>(  // NOLINT(performance-no-int-to-ptr)
            ++next_pointer
        );
        allocations.emplace(pointer, size);
        return pointer;
    }

    void copy_weight(
        void* pointer, std::size_t offset, std::span<const std::byte> source
    ) override {
        check(
            allocations.contains(pointer)
                && offset + source.size() <= allocations.at(pointer),
            "trace_fake_copy_bounds"
        );
        ++copies;
    }

    void free_weight(void* pointer) override {
        check(allocations.erase(pointer) == 1, "trace_fake_free_owned");
        ++frees;
    }

    void synchronize_weights() override {
    }
};

DeviceWeights::Key example_key() {
    constexpr auto largest = std::numeric_limits<std::uint64_t>::max();
    return {
        {largest - 100,
         largest - 99,
         largest - 98,
         largest - 97,
         largest - 96,
         largest - 95,
         largest - 94,
         largest - 93,
         largest - 92},
        largest - 91,
        8,
        4
    };
}

Json serialized_key(const DeviceWeights::Key& key) {
    const auto& source = key.source;
    return Json::array(
        {source.device,
         source.inode,
         source.file_size,
         source.modified_seconds,
         source.modified_nanoseconds,
         source.changed_seconds,
         source.changed_nanoseconds,
         source.payload_begin,
         source.payload_end,
         std::uint64_t(key.first),
         std::uint64_t(key.count),
         std::uint64_t(key.width)}
    );
}

std::vector<Json> events(const std::vector<Json>& records, const char* event) {
    std::vector<Json> result;
    for (const auto& record : records)
        if (record.at("event") == event)
            result.push_back(record);
    return result;
}

void check_sequence(const std::vector<Json>& records) {
    std::uint64_t previous = 0;
    for (const auto& record : records) {
        check(record.at("weight_cache_trace") == 1, "trace_format_version");
        const auto sequence = record.at("seq").get<std::uint64_t>();
        check(sequence > previous, "trace_sequence_increases");
        previous = sequence;
        check(record.at("bank") == records.front().at("bank"), "trace_bank_stable");
    }
}

void logical_accesses_and_identity() {
    StderrCapture capture;
    Resources resources({DeviceWeights::metadata_bytes, 1024});
    Fake device;
    DeviceWeights bank(resources, 0, Workload::llm, 64, device);
    std::atomic_bool cancel = false;
    std::array<std::byte, 32> source{};
    const auto key = example_key();
    bank.bound(64);
    check(!bank.find(key, false), "trace_public_miss");
    auto* retained = bank.retain(key, source, cancel);
    check(retained != nullptr, "trace_admission");
    check(bank.find(key, false) == retained, "trace_public_hit");
    check(bank.retain(key, source, cancel) == retained, "trace_retain_hit_uncounted");
    check(device.copies == 1, "trace_hit_no_extra_copy");
    const std::array sources{key.source};
    check(bank.release_sources(sources) == source.size(), "trace_exact_retirement");
    bank.release();
    const auto records = capture.records();
    check_sequence(records);
    const auto accesses = events(records, "access");
    check(accesses.size() == 2, "trace_two_logical_accesses");
    check(!accesses[0].at("hit").get<bool>(), "trace_miss_marked");
    check(accesses[1].at("hit").get<bool>(), "trace_hit_marked");
    for (const auto& access : accesses) {
        check(access.at("key") == serialized_key(key), "trace_full_12_field_identity");
        for (const auto& field : access.at("key"))
            check(field.is_number_unsigned(), "trace_lossless_unsigned_identity");
        check(!access.at("eligible").get<bool>(), "trace_false_eligibility");
        check(access.at("bytes") == 32, "trace_representation_bytes");
    }
    check(events(records, "open").size() == 1, "trace_open");
    check(events(records, "bound").size() == 1, "trace_bound");
    check(events(records, "admit").size() == 1, "trace_one_admit");
    const auto retirement = events(records, "retire_source");
    check(retirement.size() == 1, "trace_retirement_identity");
    auto retired_key = key;
    retired_key.first = retired_key.count = retired_key.width = 0;
    check(
        retirement[0].at("key") == serialized_key(retired_key),
        "trace_retired_source_fields"
    );
    const auto drops = events(records, "drop");
    check(
        drops.size() == 1 && drops[0].at("reason") == "retire", "trace_retirement_drop"
    );
    check(drops[0].at("allocated") == 0, "trace_drop_after_acknowledgement");
    check(events(records, "clear").size() == 1, "trace_clear");
    const auto& summary = records.back();
    check(summary.at("event") == "summary", "trace_final_summary");
    check(
        summary.at("accesses") == 2 && summary.at("hits") == 1
            && summary.at("misses") == 1,
        "trace_access_summary"
    );
    check(
        summary.at("hit_bytes") == 32 && summary.at("miss_bytes") == 32,
        "trace_byte_summary"
    );
    check(
        summary.at("admitted_bytes") == 32 && summary.at("retired_bytes") == 32,
        "trace_lifecycle_bytes"
    );
    check(
        summary.at("refused") == 0 && summary.at("dropped") == 0,
        "trace_complete_summary"
    );
    check(
        !summary.at("quarantined").get<bool>()
            && !summary.at("output_failed").get<bool>(),
        "trace_healthy_summary"
    );
    check(
        resources.snapshot().used == Footprint({0, 0}) && device.allocations.empty(),
        "trace_cleanup_ledger"
    );
}

void source_version_refusals_and_clear() {
    StderrCapture capture;
    Resources resources({DeviceWeights::metadata_bytes, 1024});
    Fake device;
    DeviceWeights bank(resources, 0, Workload::llm, 64, device);
    std::atomic_bool cancel = false;
    std::array<std::byte, 32> source{};
    auto original = example_key();
    auto changed = original;
    ++changed.source.changed_nanoseconds;
    auto refused = original;
    --refused.source.inode;
    check(bank.retain(original, source, cancel) != nullptr, "trace_original_admit");
    auto* changed_pointer = bank.retain(changed, source, cancel);
    check(changed_pointer != nullptr, "trace_changed_version_admit");
    check(!bank.retain(refused, source, cancel), "trace_envelope_refusal");
    const std::array sources{original.source};
    check(bank.release_sources(sources) == 32, "trace_only_original_retired");
    check(bank.find(changed) == changed_pointer, "trace_changed_version_preserved");
    device.free_bytes = 0;
    check(!bank.retain(refused, source, cancel), "trace_physical_refusal");
    device.free_bytes = 1024;
    device.oom = true;
    check(!bank.retain(refused, source, cancel), "trace_allocation_refusal");
    device.oom = false;
    check(bank.retain(refused, source, cancel) != nullptr, "trace_readmission");
    bank.release();
    const auto records = capture.records();
    check_sequence(records);
    const auto admissions = events(records, "admit");
    check(admissions.size() == 3, "trace_three_admissions");
    check(
        admissions[0].at("key") == serialized_key(original)
            && admissions[1].at("key") == serialized_key(changed),
        "trace_source_version_distinct"
    );
    const auto refusals = events(records, "refuse");
    check(refusals.size() == 3, "trace_three_refusals");
    check(
        refusals[0].at("reason") == "envelope" && refusals[1].at("reason") == "physical"
            && refusals[2].at("reason") == "allocation",
        "trace_refusal_reasons"
    );
    const auto drops = events(records, "drop");
    check(
        drops.size() == 3 && drops[0].at("reason") == "retire"
            && drops[1].at("reason") == "release" && drops[2].at("reason") == "release",
        "trace_release_drops"
    );
    check(
        records.back().at("refused") == 3 && records.back().at("admitted_bytes") == 96
            && records.back().at("retired_bytes") == 96
            && records.back().at("allocated") == 0,
        "trace_refusal_lifecycle_summary"
    );
    check(device.allocations.empty(), "trace_all_versions_freed");
}

void fixed_record_cap() {
    // Discard the leading bounded records; only the cap boundary reaches a
    // temporary file, so this unit never buffers or persists a huge trace.
    StderrCapture discard(true);
    Resources resources({DeviceWeights::metadata_bytes, 0, 1024});
    Fake device;
    DeviceWeights bank(resources, 1, Workload::llm, 64, device);
    const auto key = example_key();
    constexpr std::uint64_t limit = 131072;
    for (std::uint64_t i = 0; i < limit - 2; ++i)
        check(!bank.find(key), "trace_cap_miss");
    StderrCapture capture;
    check(!bank.find(key), "trace_last_permitted_record");
    check(!bank.find(key), "trace_first_truncated_record");
    check(!bank.find(key), "trace_later_truncated_record");
    bank.release();
    const auto records = capture.records();
    check(records.size() == 3, "trace_cap_constant_tail_size");
    check(
        records[0].at("event") == "access" && records[0].at("seq") == limit,
        "trace_last_normal_record"
    );
    check(records[1].at("event") == "truncated", "trace_explicit_truncation");
    check(records[2].at("event") == "summary", "trace_summary_survives_cap");
    check(
        records[2].at("emitted") == limit && records[2].at("dropped") == 3,
        "trace_exact_cap_accounting"
    );
    check(
        records[2].at("accesses") == limit + 1 && records[2].at("misses") == limit + 1,
        "trace_uncapped_access_totals"
    );
    check_sequence(records);
}

void fixed_summary_cap() {
    // A separate fake device keeps this quota independent from the ordinary
    // record-cap test. Discard warmup summaries instead of storing them.
    StderrCapture discard(true);
    Resources resources({DeviceWeights::metadata_bytes, 0, 0, 1024});
    Fake device;
    DeviceWeights first_bank(resources, 2, Workload::llm, 64, device);
    constexpr unsigned limit = 1024;
    for (unsigned i = 0; i < limit - 1; ++i)
        first_bank.release();
    StderrCapture capture;
    // The summary quota belongs to the device, so a new bank cannot reset it.
    DeviceWeights second_bank(resources, 2, Workload::llm, 64, device);
    second_bank.release();
    second_bank.release();
    second_bank.release();
    const auto records = capture.records();
    check_sequence(records);
    check(records.size() == 6, "trace_summary_cap_constant_tail_size");
    const auto summaries = events(records, "summary");
    check(summaries.size() == 1, "trace_last_allowed_summary_only");
    check(
        summaries[0].at("accesses") == 0 && summaries[0].at("allocated") == 0,
        "trace_last_summary_valid"
    );
    const auto truncated = events(records, "truncated");
    check(
        truncated.size() == 1 && truncated[0].at("reason") == "summary_limit",
        "trace_summary_explicit_truncation_once"
    );
    check(
        events(records, "clear").size() == 3 && records.back().at("event") == "clear",
        "trace_summary_suppression_preserves_clear"
    );
    check(
        summaries[0].at("seq").get<std::uint64_t>()
            < truncated[0].at("seq").get<std::uint64_t>(),
        "trace_summary_notice_after_last_summary"
    );
    check(
        device.allocations.empty()
            && resources.snapshot().used == Footprint({0, 0, 0, 0}),
        "trace_summary_cap_no_allocations"
    );
}
}  // namespace

int main() {
    try {
        logical_accesses_and_identity();
        source_version_refusals_and_clear();
        fixed_record_cap();
        fixed_summary_cap();
        std::cout << "device weight trace tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
