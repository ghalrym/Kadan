#include "kadan/platform.hpp"

#ifndef _WIN32

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <utility>

namespace kadan::platform {
namespace {
constexpr std::uint64_t kHostMargin = 1024ULL * 1024 * 1024;
}  // namespace

std::string machine_boot_id() {
    std::string boot;
    std::ifstream("/proc/sys/kernel/random/boot_id") >> boot;
    if (boot.empty())
        throw std::runtime_error("computer_identity");
    return boot;
}

std::string host_name() {
    std::array<char, 256> host{};
    if (gethostname(host.data(), host.size() - 1) != 0)
        return "localhost";
    return host.data();
}

std::uint64_t process_id() {
    return static_cast<std::uint64_t>(getpid());
}

std::filesystem::path executable_path() {
    std::array<char, 4096> executable{};
    const auto size = readlink("/proc/self/exe", executable.data(), executable.size());
    if (size > 0 && std::size_t(size) < executable.size())
        return std::filesystem::path(std::string(executable.data(), std::size_t(size)));
    return {};
}

MemoryInfo probe_memory(bool apply_container_limit) {
    MemoryInfo info{};
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    std::uint64_t avail = 0;
    while (std::getline(meminfo, line)) {
        if (line.starts_with("MemTotal:"))
            info.physical_ram_total = std::stoull(line.substr(9)) * 1024ULL;
        else if (line.starts_with("MemAvailable:"))
            avail = std::stoull(line.substr(13)) * 1024ULL;
    }
    if (avail == 0)
        throw std::runtime_error("host_memory_probe");
    info.ram_total = info.physical_ram_total;
    info.physical_ram_free = avail > kHostMargin ? avail - kHostMargin : 0;

    std::uint64_t effective_avail = avail;
    std::ifstream limit_file("/sys/fs/cgroup/memory.max");
    std::ifstream current_file("/sys/fs/cgroup/memory.current");
    std::string limit;
    std::uint64_t current = 0;
    if (limit_file >> limit && limit != "max") {
        const std::uint64_t maximum = std::stoull(limit);
        info.ram_limit = maximum > kHostMargin ? maximum - kHostMargin : 0;
        if (apply_container_limit) {
            info.ram_total = std::min<std::uint64_t>(info.ram_total, maximum);
            if (current_file >> current) {
                const std::uint64_t cgroup_avail =
                    maximum > current ? maximum - current : 0;
                effective_avail = std::min<std::uint64_t>(effective_avail, cgroup_avail);
            }
        }
    }
    info.ram_free = effective_avail > kHostMargin ? effective_avail - kHostMargin : 0;
    return info;
}

DiskInfo probe_disk(const std::filesystem::path& path) {
    struct statvfs disk{};
    if (statvfs(path.c_str(), &disk) != 0)
        throw std::runtime_error("disk_probe");
    return {
        static_cast<std::uint64_t>(disk.f_blocks) * disk.f_frsize,
        static_cast<std::uint64_t>(disk.f_bavail) * disk.f_frsize
    };
}

std::uint64_t process_rss_bytes() {
    std::ifstream stat("/proc/self/statm");
    std::uint64_t size = 0, resident = 0;
    const auto page = sysconf(_SC_PAGESIZE);
    if (!(stat >> size >> resident) || page <= 0)
        return 0;
    return resident * static_cast<std::uint64_t>(page);
}

DynamicLibrary::DynamicLibrary(const char* name) {
    if (name && *name)
        handle_ = dlopen(name, RTLD_NOW);
}

DynamicLibrary::~DynamicLibrary() {
    if (handle_)
        dlclose(handle_);
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)) {
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        if (handle_)
            dlclose(handle_);
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

void* DynamicLibrary::raw_symbol(const char* name) const {
    return handle_ ? dlsym(handle_, name) : nullptr;
}

const char* nvml_library_name() {
    return "libnvidia-ml.so.1";
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
    const int fd =
        open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0)
        throw std::runtime_error("file_lock_open");
    const int op = LOCK_EX | (non_blocking ? LOCK_NB : 0);
    if (flock(fd, op) != 0) {
        const int err = errno;
        close(fd);
        if (non_blocking && (err == EWOULDBLOCK || err == EAGAIN))
            return false;
        throw std::runtime_error("file_lock_failed");
    }
    handle_ = fd;
    locked_ = true;
    return true;
}

void FileLock::release() {
    if (handle_ >= 0) {
        close(static_cast<int>(handle_));
        handle_ = -1;
    }
    locked_ = false;
}

std::filesystem::path default_model_dir() {
    const char* home = std::getenv("HOME");
    return std::filesystem::path(home && *home ? home : "/tmp")
           / ".local/share/kadan/models";
}

}  // namespace kadan::platform

#endif  // !_WIN32
