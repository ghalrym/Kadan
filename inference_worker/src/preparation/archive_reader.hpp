#pragma once
#include <archive.h>
#include <archive_entry.h>
#include <memory>

#include "kadan/preparation.hpp"

namespace kadan::serving {
class ArchiveReader {
public:
    explicit ArchiveReader(const fs::path& path)
        : archive_(archive_read_new(), archive_read_free) {
        preparation_check(bool(archive_), "Cannot allocate checkpoint archive reader");
        archive_read_support_format_zip(archive_.get());
        preparation_check(
            archive_read_open_filename(archive_.get(), path.c_str(), 1024 * 1024)
                == ARCHIVE_OK,
            "Cannot open checkpoint archive"
        );
    }

    bool next() {
        const auto result = archive_read_next_header(archive_.get(), &entry_);
        preparation_check(
            result == ARCHIVE_OK || result == ARCHIVE_EOF, "Invalid checkpoint archive"
        );
        return result == ARCHIVE_OK;
    }

    std::string name() const {
        return archive_entry_pathname(entry_);
    }

    std::string read(std::size_t maximum, const std::atomic_bool& cancel) {
        const auto size = archive_entry_size(entry_);
        preparation_check(
            size >= 0 && std::uint64_t(size) <= maximum
                && archive_entry_filetype(entry_) == AE_IFREG,
            "Checkpoint archive entry bound"
        );
        std::string result(size, '\0');
        std::size_t offset = 0;
        while (offset < result.size()) {
            preparation_cancel(cancel);
            const auto count = archive_read_data(
                archive_.get(),
                result.data() + offset,
                std::min<std::size_t>(1024 * 1024, result.size() - offset)
            );
            preparation_check(count > 0, "Truncated checkpoint archive entry");
            offset += count;
        }
        return result;
    }

private:
    std::unique_ptr<struct archive, decltype(&archive_read_free)> archive_;
    struct archive_entry* entry_ = nullptr;
};
}  // namespace kadan::serving
