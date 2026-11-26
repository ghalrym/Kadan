#include "kadan/platform.hpp"

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include <array>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kadan::platform {
namespace {
constexpr std::uint64_t kHostMargin = 1024ULL * 1024 * 1024;
}  // namespace

std::string machine_boot_id() {
    std::string guid;
    HKEY key = nullptr;
    if (RegOpenKeyExA(
            HKEY_LOCAL_MACHINE,
            "SOFTWARE\\Microsoft\\Cryptography",
            0,
            KEY_READ | KEY_WOW64_64KEY,
            &key
        )
        == ERROR_SUCCESS) {
        char buffer[128]{};
        DWORD size = sizeof(buffer);
        if (RegQueryValueExA(
                key,
                "MachineGuid",
                nullptr,
                nullptr,
                reinterpret_cast<LPBYTE>(buffer),
                &size
            )
            == ERROR_SUCCESS) {
            guid = buffer;
        }
        RegCloseKey(key);
    }
    if (guid.empty())
        guid = host_name();
    return "win-" + guid;
}

std::string host_name() {
    std::array<char, MAX_COMPUTERNAME_LENGTH + 1> buffer{};
    DWORD size = static_cast<DWORD>(buffer.size());
    if (!GetComputerNameA(buffer.data(), &size))
        return "localhost";
    return std::string(buffer.data(), size);
}

std::uint64_t process_id() {
    return static_cast<std::uint64_t>(GetCurrentProcessId());
}

std::filesystem::path executable_path() {
    std::array<wchar_t, 4096> buffer{};
    const DWORD length =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length > 0 && length < buffer.size())
        return std::filesystem::path(std::wstring(buffer.data(), length));
    return {};
}

MemoryInfo probe_memory(bool /*apply_container_limit*/) {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status))
        throw std::runtime_error("host_memory_probe");
    MemoryInfo info{};
    info.physical_ram_total = static_cast<std::uint64_t>(status.ullTotalPhys);
    info.ram_total = info.physical_ram_total;
    const auto avail = static_cast<std::uint64_t>(status.ullAvailPhys);
    info.physical_ram_free = avail > kHostMargin ? avail - kHostMargin : 0;
    info.ram_free = info.physical_ram_free;
    return info;
}

DiskInfo probe_disk(const std::filesystem::path& path) {
    ULARGE_INTEGER available{}, total{}, total_free{};
    if (!GetDiskFreeSpaceExW(path.c_str(), &available, &total, &total_free))
        throw std::runtime_error("disk_probe");
    return {
        static_cast<std::uint64_t>(total.QuadPart),
        static_cast<std::uint64_t>(available.QuadPart)
    };
}

std::uint64_t process_rss_bytes() {
    PROCESS_MEMORY_COUNTERS counters{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
        return 0;
    return static_cast<std::uint64_t>(counters.WorkingSetSize);
}

DynamicLibrary::DynamicLibrary(const char* name) {
    if (name && *name)
        handle_ = reinterpret_cast<void*>(LoadLibraryA(name));
}

DynamicLibrary::~DynamicLibrary() {
    if (handle_)
        FreeLibrary(reinterpret_cast<HMODULE>(handle_));
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)) {
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        if (handle_)
            FreeLibrary(reinterpret_cast<HMODULE>(handle_));
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

void* DynamicLibrary::raw_symbol(const char* name) const {
    return handle_ ? reinterpret_cast<void*>(
                         GetProcAddress(reinterpret_cast<HMODULE>(handle_), name)
                     )
                   : nullptr;
}

const char* nvml_library_name() {
    return "nvml.dll";
}

FileLock::~FileLock() {
    release();
}

FileLock::FileLock(FileLock&& other) noexcept
    : handle_(std::exchange(other.handle_, -1)),
      locked_(std::exchange(other.locked_, false)) {
}

FileLock& FileLock::operator=(FileLock&& other) noexcept {
    if (this != &other) {
        release();
        handle_ = std::exchange(other.handle_, -1);
        locked_ = std::exchange(other.locked_, false);
    }
    return *this;
}

bool FileLock::acquire(const std::filesystem::path& path, bool non_blocking) {
    release();
    HANDLE file = CreateFileW(
        path.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("file_lock_open");
    OVERLAPPED overlapped{};
    DWORD flags = LOCKFILE_EXCLUSIVE_LOCK;
    if (non_blocking)
        flags |= LOCKFILE_FAIL_IMMEDIATELY;
    if (!LockFileEx(file, flags, 0, MAXDWORD, MAXDWORD, &overlapped)) {
        const DWORD err = GetLastError();
        CloseHandle(file);
        if (non_blocking && err == ERROR_LOCK_VIOLATION)
            return false;
        throw std::runtime_error("file_lock_failed");
    }
    handle_ = reinterpret_cast<std::intptr_t>(file);
    locked_ = true;
    return true;
}

void FileLock::release() {
    if (handle_ != -1) {
        HANDLE file = reinterpret_cast<HANDLE>(handle_);
        OVERLAPPED overlapped{};
        UnlockFileEx(file, 0, MAXDWORD, MAXDWORD, &overlapped);
        CloseHandle(file);
        handle_ = -1;
    }
    locked_ = false;
}

std::filesystem::path default_model_dir() {
    const char* local_app_data = std::getenv("LOCALAPPDATA");
    if (local_app_data && *local_app_data)
        return std::filesystem::path(local_app_data) / "Kadan" / "models";
    const char* profile = std::getenv("USERPROFILE");
    return std::filesystem::path(profile && *profile ? profile : ".") / "Kadan"
           / "models";
}

}  // namespace kadan::platform

#endif  // _WIN32
