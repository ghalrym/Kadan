#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace kadan::platform {

struct MemoryInfo {
    std::uint64_t ram_total = 0;
    std::uint64_t ram_free = 0;
    std::uint64_t physical_ram_total = 0;
    std::uint64_t physical_ram_free = 0;
    std::optional<std::uint64_t> ram_limit;
};

struct DiskInfo {
    std::uint64_t total = 0;
    std::uint64_t available = 0;
};

// Stable computer identity shared by all workers on the same host.
std::string machine_boot_id();

// Local host name.
std::string host_name();

// Current process ID.
std::uint64_t process_id();

// Absolute path to the current executable, or empty on failure.
std::filesystem::path executable_path();

// Host RAM and optional container cgroup memory limit (with 1 GiB headroom removed
// from free/limit values, matching Hikari's memory ledger contract).
MemoryInfo probe_memory(bool apply_container_limit = true);

// Filesystem capacity and available space for the given path.
DiskInfo probe_disk(const std::filesystem::path& path);

// Current process resident set size in bytes (0 if unavailable).
std::uint64_t process_rss_bytes();

// Dynamic library loader for NVML (libnvidia-ml.so.1 on Linux, nvml.dll on Windows).
class DynamicLibrary {
public:
    DynamicLibrary() = default;
    explicit DynamicLibrary(const char* name);
    ~DynamicLibrary();
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    DynamicLibrary(DynamicLibrary&& other) noexcept;
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

    explicit operator bool() const noexcept {
        return handle_ != nullptr;
    }

    void* raw_symbol(const char* name) const;

    template <class Fn> Fn symbol(const char* name) const {
        return reinterpret_cast<Fn>(raw_symbol(name));
    }

private:
    void* handle_ = nullptr;
};

// Returns the platform NVML shared library filename.
const char* nvml_library_name();

// Exclusive file lock (non-blocking or blocking). Automatically unlocked on destruction.
class FileLock {
public:
    FileLock() = default;
    ~FileLock();
    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;
    FileLock(FileLock&& other) noexcept;
    FileLock& operator=(FileLock&& other) noexcept;

    // Attempts to acquire an exclusive lock on `path`. Throws on OS error; returns
    // false if non_blocking is true and the lock is already held by another process.
    bool acquire(const std::filesystem::path& path, bool non_blocking);
    void release();

    explicit operator bool() const noexcept {
        return locked_;
    }

private:
    std::intptr_t handle_ = -1;
    bool locked_ = false;
};

// Default user data directory for models when KADAN_MODEL_DIR is unset.
std::filesystem::path default_model_dir();

}  // namespace kadan::platform
