#include "kadan/gguf_checkpoint.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <ggml.h>
#include <gguf.h>
#include <limits>
#include <regex>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <utility>

namespace kadan::checkpoint {
namespace {
constexpr std::uint64_t metadata_limit = 32 * 1024 * 1024;

void require(bool valid, const char* message) {
    if (!valid)
        throw std::invalid_argument(message);
}

void stop(const std::atomic_bool& cancel) {
    if (cancel.load())
        throw std::runtime_error("gguf_cancelled");
}

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    require(b <= UINT64_MAX - a, "gguf_size_overflow");
    return a + b;
}

std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    require(!b || a <= UINT64_MAX / b, "gguf_size_overflow");
    return a * b;
}

struct File {
    int fd;
    struct stat initial{};

    File(int descriptor, bool directory = false) : fd(descriptor) {
        if (fd < 0)
            throw std::runtime_error("gguf_open_failed");
        if (::fstat(fd, &initial)
            || (directory ? !S_ISDIR(initial.st_mode)
                          : (!S_ISREG(initial.st_mode) || initial.st_size < 24))) {
            ::close(fd);
            fd = -1;
            throw std::invalid_argument("gguf_not_regular_file");
        }
    }

    File(File&& other) noexcept
        : fd(std::exchange(other.fd, -1)), initial(other.initial) {
    }

    File(const File&) = delete;

    ~File() {
        if (fd >= 0)
            ::close(fd);
    }

    void unchanged() const {
        struct stat now{};
        require(
            !::fstat(fd, &now) && now.st_size == initial.st_size
                && now.st_mtim.tv_sec == initial.st_mtim.tv_sec
                && now.st_mtim.tv_nsec == initial.st_mtim.tv_nsec
                && now.st_ctim.tv_sec == initial.st_ctim.tv_sec
                && now.st_ctim.tv_nsec == initial.st_ctim.tv_nsec,
            "gguf_checkpoint_changed"
        );
    }

    void read(std::uint64_t at, std::span<std::uint8_t> output) const {
        require(
            at <= std::uint64_t(initial.st_size)
                && output.size() <= std::uint64_t(initial.st_size) - at,
            "gguf_read_bounds"
        );
        while (!output.empty()) {
            const auto n =
                ::pread(fd, output.data(), output.size(), static_cast<off_t>(at));
            if (n < 0 && errno == EINTR)
                continue;
            if (n <= 0)
                throw std::runtime_error("gguf_read_failed");
            at += std::size_t(n);
            output = output.subspan(std::size_t(n));
        }
    }
};

struct ContextDeleter {
    void operator()(gguf_context* context) const {
        gguf_free(context);
    }
};

using Context = std::unique_ptr<gguf_context, ContextDeleter>;

// Preflight before invoking upstream GGUF parsing: bound every length/count and
// validate types/dimensions before ggml helpers that assume valid enum values.
class Header {
public:
    const File& file;
    const std::atomic_bool& cancel;
    std::uint64_t at = 0;
    std::vector<GgufTensor> tensors;
    std::vector<std::uint64_t> offsets;

    Header(const File& f, const std::atomic_bool& c) : file(f), cancel(c) {
        require(u32() == 0x46554747 && u32() == 3, "gguf_version");
        const auto tensor_count = integer(), kv_count = integer();
        require(tensor_count <= 200000 && kv_count <= 16384, "gguf_header_count");
        std::unordered_set<std::string> keys;
        for (std::uint64_t i = 0; i < kv_count; ++i) {
            auto key = text(256);
            require(keys.insert(key).second, "gguf_duplicate_key");
            const auto type = u32();
            if (key == "general.alignment") {
                require(type == GGUF_TYPE_UINT32, "gguf_alignment");
                const auto alignment = u32();
                require(
                    alignment && alignment <= 4096 && !(alignment & (alignment - 1)),
                    "gguf_alignment"
                );
            } else
                value(type);
        }
        std::unordered_set<std::string> names;
        for (std::uint64_t i = 0; i < tensor_count; ++i) {
            GgufTensor tensor;
            tensor.name = text(255);
            require(
                !tensor.name.empty() && names.insert(tensor.name).second,
                "gguf_duplicate_tensor"
            );
            tensor.rank = u32();
            require(tensor.rank >= 1 && tensor.rank <= 3, "gguf_tensor_rank");
            std::uint64_t rows = 1;
            for (std::size_t d = 0; d < tensor.rank; ++d) {
                tensor.shape[d] = integer();
                require(
                    tensor.shape[d] && tensor.shape[d] <= INT64_MAX, "gguf_tensor_shape"
                );
                if (d)
                    rows = mul(rows, tensor.shape[d]);
            }
            tensor.type = u32();
            require(
                tensor.type == GGML_TYPE_F32 || tensor.type == GGML_TYPE_Q8_0
                    || tensor.type == GGML_TYPE_Q3_K || tensor.type == GGML_TYPE_Q4_K,
                "gguf_tensor_type"
            );
            const auto type = static_cast<ggml_type>(tensor.type);
            require(
                tensor.shape[0] % ggml_blck_size(type) == 0,
                "gguf_tensor_block_alignment"
            );
            require(
                mul(tensor.shape[0], rows) <= INT64_MAX
                    && tensor.shape[0] <= SIZE_MAX / ggml_type_size(type),
                "gguf_tensor_shape"
            );
            tensor.bytes = mul(ggml_row_size(type, tensor.shape[0]), rows);
            require(tensor.bytes <= INT64_MAX, "gguf_tensor_shape");
            offsets.push_back(integer());
            tensors.push_back(std::move(tensor));
        }
    }

private:
    void bytes(std::span<std::uint8_t> output) {
        stop(cancel);
        require(
            at <= metadata_limit && output.size() <= metadata_limit - at,
            "gguf_metadata_limit"
        );
        file.read(at, output);
        at += output.size();
    }

    void skip(std::uint64_t size) {
        stop(cancel);
        require(
            at <= metadata_limit && size <= metadata_limit - at
                && size <= std::uint64_t(file.initial.st_size) - at,
            "gguf_metadata_limit"
        );
        at += size;
    }

    std::uint64_t integer() {
        std::array<std::uint8_t, 8> data{};
        bytes(data);
        std::uint64_t result = 0;
        for (unsigned i = 0; i < 8; ++i)
            result |= std::uint64_t(data[i]) << (8 * i);
        return result;
    }

    std::uint32_t u32() {
        std::array<std::uint8_t, 4> data{};
        bytes(data);
        std::uint32_t result = 0;
        for (unsigned i = 0; i < 4; ++i)
            result |= std::uint32_t(data[i]) << (8 * i);
        return result;
    }

    std::string text(std::size_t limit) {
        const auto length = integer();
        require(length <= limit, "gguf_string_limit");
        std::string result(length, '\0');
        bytes({reinterpret_cast<std::uint8_t*>(result.data()), result.size()});
        require(result.find('\0') == std::string::npos, "gguf_string_nul");
        return result;
    }

    void value(std::uint32_t type) {
        if (type == GGUF_TYPE_STRING) {
            (void)text(1024 * 1024);
            return;
        }
        if (type == GGUF_TYPE_ARRAY) {
            const auto element = u32();
            const auto count = integer();
            require(count <= 524288 && element != GGUF_TYPE_ARRAY, "gguf_array_limit");
            if (element == GGUF_TYPE_STRING) {
                for (std::uint64_t i = 0; i < count; ++i)
                    (void)text(1024 * 1024);
            } else
                skip(mul(count, width(element)));
            return;
        }
        skip(width(type));
    }

    static std::size_t width(std::uint32_t type) {
        switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        default:
            throw std::invalid_argument("gguf_metadata_type");
        }
    }
};

struct Reader {
    const File& file;
    std::uint64_t end;
    const std::atomic_bool& cancel;
    bool failed = false;
};

std::size_t read_metadata(void* opaque, void* output, std::uint64_t at, std::size_t n) {
    auto& reader = *static_cast<Reader*>(opaque);
    if (reader.cancel.load() || at > reader.end || n > reader.end - at)
        return 0;
    try {
        reader.file.read(at, {static_cast<std::uint8_t*>(output), n});
        return n;
    } catch (...) {
        reader.failed = true;
        return 0;
    }
}

Context parse_metadata(const File& file, const Header& header) {
    Reader reader{file, header.at, header.cancel};
    Context context(gguf_init_from_callback(
        read_metadata, &reader, 65536, file.initial.st_size, {true, nullptr}
    ));
    stop(header.cancel);
    require(context && !reader.failed, "gguf_invalid_metadata");
    file.unchanged();
    return context;
}

std::int64_t key_id(const gguf_context* ctx, std::string_view name) {
    require(
        name.size() <= 256 && name.find('\0') == std::string_view::npos,
        "gguf_metadata_key"
    );
    const auto id = gguf_find_key(ctx, std::string(name).c_str());
    require(id >= 0, "gguf_missing_metadata");
    return id;
}

std::uint64_t integer(const gguf_context* ctx, std::int64_t id) {
    switch (gguf_get_kv_type(ctx, id)) {
    case GGUF_TYPE_UINT8:
        return gguf_get_val_u8(ctx, id);
    case GGUF_TYPE_UINT16:
        return gguf_get_val_u16(ctx, id);
    case GGUF_TYPE_UINT32:
        return gguf_get_val_u32(ctx, id);
    case GGUF_TYPE_UINT64:
        return gguf_get_val_u64(ctx, id);
    case GGUF_TYPE_INT32: {
        const auto v = gguf_get_val_i32(ctx, id);
        require(v >= 0, "gguf_metadata_integer");
        return v;
    }
    case GGUF_TYPE_INT64: {
        const auto v = gguf_get_val_i64(ctx, id);
        require(v >= 0, "gguf_metadata_integer");
        return v;
    }
    default:
        throw std::invalid_argument("gguf_metadata_integer");
    }
}
}  // namespace

struct GgufCheckpoint::Impl {
    struct Entry {
        GgufTensor tensor;
        std::size_t file;
        std::uint64_t begin;
    };

    std::vector<File> files;
    std::vector<Entry> tensors;
    Context metadata;

    const Entry& find(std::string_view name) const {
        auto it = std::lower_bound(
            tensors.begin(),
            tensors.end(),
            name,
            [](const Entry& e, std::string_view n) { return e.tensor.name < n; }
        );
        require(it != tensors.end() && it->tensor.name == name, "gguf_missing_tensor");
        return *it;
    }
};

GgufCheckpoint::GgufCheckpoint(const std::string& root, const std::atomic_bool& cancel)
    : impl_(std::make_unique<Impl>()) {
    require(std::endian::native == std::endian::little, "gguf_host_endian");
    const std::regex pattern(R"((.+)-([0-9]{5})-of-([0-9]{5})\.gguf)");
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        stop(cancel);
        if (entry.path().extension() == ".gguf")
            names.push_back(entry.path().filename().string());
        require(names.size() <= 512, "gguf_split_count");
    }
    require(!names.empty(), "gguf_missing_files");
    std::sort(names.begin(), names.end());
    std::string prefix;
    std::uint64_t total_tensors = 0, metadata_bytes = 0;
    File directory(
        ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC), true
    );
    for (std::size_t i = 0; i < names.size(); ++i) {
        std::smatch match;
        require(
            names[i].size() <= 255 && std::regex_match(names[i], match, pattern),
            "gguf_split_name"
        );
        if (i == 0)
            prefix = match[1];
        require(
            match[1] == prefix && std::stoul(match[2]) == i + 1
                && std::stoul(match[3]) == names.size(),
            "gguf_split_name"
        );
        File opened(
            ::openat(directory.fd, names[i].c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        );
        impl_->files.push_back(std::move(opened));
        const auto& file = impl_->files.back();
        Header header(file, cancel);
        metadata_bytes = add(metadata_bytes, header.at);
        require(metadata_bytes <= 64 * 1024 * 1024, "gguf_metadata_limit");
        auto context = parse_metadata(file, header);
        require(gguf_get_version(context.get()) == 3, "gguf_version");
        require(
            ::kadan::checkpoint::integer(
                context.get(), key_id(context.get(), "split.no")
            ) == i
                && ::kadan::checkpoint::integer(
                       context.get(), key_id(context.get(), "split.count")
                   ) == names.size(),
            "gguf_split_metadata"
        );
        const auto expected = ::kadan::checkpoint::integer(
            context.get(), key_id(context.get(), "split.tensors.count")
        );
        if (i == 0)
            total_tensors = expected;
        require(
            expected == total_tensors && total_tensors <= 200000, "gguf_split_metadata"
        );
        require(
            gguf_get_n_tensors(context.get()) == std::int64_t(header.tensors.size()),
            "gguf_tensor_count"
        );
        const auto begin = gguf_get_data_offset(context.get());
        const auto alignment = gguf_get_alignment(context.get());
        require(
            alignment && alignment <= 4096 && !(alignment & (alignment - 1)),
            "gguf_alignment"
        );
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (std::size_t t = 0; t < header.tensors.size(); ++t) {
            const auto offset = header.offsets[t];
            const auto end = add(add(begin, offset), header.tensors[t].bytes);
            require(
                offset % alignment == 0 && end <= std::uint64_t(file.initial.st_size),
                "gguf_tensor_bounds"
            );
            require(
                header.tensors[t].name == gguf_get_tensor_name(context.get(), t)
                    && offset == gguf_get_tensor_offset(context.get(), t)
                    && header.tensors[t].bytes
                           == gguf_get_tensor_size(context.get(), t),
                "gguf_tensor_layout"
            );
            ranges.emplace_back(offset, add(offset, header.tensors[t].bytes));
            impl_->tensors.push_back(
                {std::move(header.tensors[t]), i, add(begin, offset)}
            );
        }
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t t = 1; t < ranges.size(); ++t)
            require(ranges[t].first >= ranges[t - 1].second, "gguf_tensor_overlap");
        file.unchanged();
        if (i == 0)
            impl_->metadata = std::move(context);
    }
    require(impl_->tensors.size() == total_tensors, "gguf_tensor_count");
    std::sort(
        impl_->tensors.begin(), impl_->tensors.end(), [](const auto& a, const auto& b) {
            return a.tensor.name < b.tensor.name;
        }
    );
    for (std::size_t i = 1; i < impl_->tensors.size(); ++i)
        require(
            impl_->tensors[i - 1].tensor.name != impl_->tensors[i].tensor.name,
            "gguf_duplicate_tensor"
        );
    check_unchanged();
}

GgufCheckpoint::~GgufCheckpoint() = default;

std::size_t GgufCheckpoint::tensor_count() const {
    return impl_->tensors.size();
}

const GgufTensor& GgufCheckpoint::tensor(std::string_view name) const {
    return impl_->find(name).tensor;
}

const GgufTensor& GgufCheckpoint::tensor_at(std::size_t index) const {
    if (index >= impl_->tensors.size())
        throw std::out_of_range("gguf_tensor_index");
    return impl_->tensors[index].tensor;
}

void GgufCheckpoint::check_unchanged() const {
    for (const auto& file : impl_->files)
        file.unchanged();
}

WeightIdentity
GgufCheckpoint::identity(std::string_view name, std::size_t expert) const {
    const auto& entry = impl_->find(name);
    const auto& t = entry.tensor;
    require(expert < t.shape[2], "gguf_expert_range");
    const auto& file = impl_->files[entry.file];
    file.unchanged();
    const auto& s = file.initial;
    const auto bytes = t.bytes / t.shape[2];
    const auto begin = add(entry.begin, mul(expert, bytes));
    return {
        std::uint64_t(s.st_dev),
        std::uint64_t(s.st_ino),
        std::uint64_t(s.st_size),
        std::uint64_t(s.st_mtim.tv_sec),
        std::uint64_t(s.st_mtim.tv_nsec),
        std::uint64_t(s.st_ctim.tv_sec),
        std::uint64_t(s.st_ctim.tv_nsec),
        begin,
        add(begin, bytes)
    };
}

void GgufCheckpoint::read_rows(
    std::string_view name,
    std::size_t expert,
    std::size_t first_row,
    std::span<float> output,
    const std::atomic_bool& cancel
) const {
    stop(cancel);
    const auto& entry = impl_->find(name);
    const auto& t = entry.tensor;
    require(
        t.rank >= 2 && expert < t.shape[2] && output.size() % t.shape[0] == 0
            && first_row <= t.shape[1]
            && output.size() / t.shape[0] <= t.shape[1] - first_row,
        "gguf_row_range"
    );
    const auto type = static_cast<ggml_type>(t.type);
    const auto row_bytes = ggml_row_size(type, t.shape[0]);
    require(row_bytes <= 1024 * 1024, "gguf_row_scratch_limit");
    std::vector<std::uint8_t> packed(row_bytes);
    const auto& file = impl_->files[entry.file];
    file.unchanged();
    const auto first = add(mul(expert, t.shape[1]), first_row);
    for (std::size_t r = 0; r < output.size() / t.shape[0]; ++r) {
        stop(cancel);
        file.read(add(entry.begin, mul(add(first, r), row_bytes)), packed);
        auto row = output.subspan(r * t.shape[0], t.shape[0]);
        if (type == GGML_TYPE_F32)
            std::memcpy(row.data(), packed.data(), row_bytes);
        else
            ggml_get_type_traits(type)->to_float(packed.data(), row.data(), t.shape[0]);
        for (auto value : row)
            require(std::isfinite(value), "gguf_nonfinite_weight");
    }
    file.unchanged();
    stop(cancel);
}

void GgufCheckpoint::read_vector(
    std::string_view name, std::span<float> output, const std::atomic_bool& cancel
) const {
    stop(cancel);
    const auto& entry = impl_->find(name);
    const auto& t = entry.tensor;
    require(
        t.rank == 1 && t.type == GGML_TYPE_F32 && output.size() == t.shape[0],
        "gguf_vector_shape"
    );
    const auto& file = impl_->files[entry.file];
    file.unchanged();
    auto raw =
        std::span(reinterpret_cast<std::uint8_t*>(output.data()), output.size_bytes());
    for (std::size_t at = 0; at < raw.size();) {
        stop(cancel);
        const auto count = std::min<std::size_t>(65536, raw.size() - at);
        file.read(add(entry.begin, at), raw.subspan(at, count));
        at += count;
    }
    for (auto value : output)
        require(std::isfinite(value), "gguf_nonfinite_weight");
    file.unchanged();
    stop(cancel);
}

namespace {
std::string metadata_string(const gguf_context* ctx, std::string_view key) {
    const auto id = key_id(ctx, key);
    require(gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING, "gguf_metadata_string");
    return gguf_get_val_str(ctx, id);
}

double metadata_number(const gguf_context* ctx, std::string_view key) {
    const auto id = key_id(ctx, key);
    const auto type = gguf_get_kv_type(ctx, id);
    const double value = type == GGUF_TYPE_FLOAT32   ? gguf_get_val_f32(ctx, id)
                         : type == GGUF_TYPE_FLOAT64 ? gguf_get_val_f64(ctx, id)
                                                     : double(integer(ctx, id));
    require(std::isfinite(value), "gguf_metadata_number");
    return value;
}

bool metadata_boolean(const gguf_context* ctx, std::string_view key) {
    const auto id = key_id(ctx, key);
    require(gguf_get_kv_type(ctx, id) == GGUF_TYPE_BOOL, "gguf_metadata_boolean");
    return gguf_get_val_bool(ctx, id);
}

std::vector<bool> metadata_booleans(const gguf_context* ctx, std::string_view key) {
    const auto id = key_id(ctx, key);
    require(
        gguf_get_kv_type(ctx, id) == GGUF_TYPE_ARRAY
            && gguf_get_arr_type(ctx, id) == GGUF_TYPE_BOOL,
        "gguf_metadata_booleans"
    );
    const auto n = gguf_get_arr_n(ctx, id);
    require(n <= 4096, "gguf_metadata_booleans");
    const auto* raw = static_cast<const std::uint8_t*>(gguf_get_arr_data(ctx, id));
    std::vector<bool> result;
    result.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        require(raw[i] <= 1, "gguf_metadata_booleans");
        result.push_back(raw[i]);
    }
    return result;
}
}  // namespace

std::string GgufCheckpoint::string(std::string_view key) const {
    return metadata_string(impl_->metadata.get(), key);
}

std::uint64_t GgufCheckpoint::integer(std::string_view key) const {
    const auto* ctx = impl_->metadata.get();
    return ::kadan::checkpoint::integer(ctx, key_id(ctx, key));
}

double GgufCheckpoint::number(std::string_view key) const {
    return metadata_number(impl_->metadata.get(), key);
}

bool GgufCheckpoint::boolean(std::string_view key) const {
    return metadata_boolean(impl_->metadata.get(), key);
}

std::vector<bool> GgufCheckpoint::booleans(std::string_view key) const {
    return metadata_booleans(impl_->metadata.get(), key);
}

struct GgufMetadata::Impl {
    Context metadata;
};

GgufMetadata::GgufMetadata(const std::string& path, const std::atomic_bool& cancel)
    : impl_(std::make_unique<Impl>()) {
    require(std::endian::native == std::endian::little, "gguf_host_endian");
    File file(::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
    Header header(file, cancel);
    impl_->metadata = parse_metadata(file, header);
    stop(cancel);
}

GgufMetadata::~GgufMetadata() = default;

std::size_t GgufMetadata::tensor_count() const {
    return gguf_get_n_tensors(impl_->metadata.get());
}

std::string GgufMetadata::string(std::string_view key) const {
    return metadata_string(impl_->metadata.get(), key);
}

std::uint64_t GgufMetadata::integer(std::string_view key) const {
    const auto* ctx = impl_->metadata.get();
    return ::kadan::checkpoint::integer(ctx, key_id(ctx, key));
}

double GgufMetadata::number(std::string_view key) const {
    return metadata_number(impl_->metadata.get(), key);
}

bool GgufMetadata::boolean(std::string_view key) const {
    return metadata_boolean(impl_->metadata.get(), key);
}

std::vector<bool> GgufMetadata::booleans(std::string_view key) const {
    return metadata_booleans(impl_->metadata.get(), key);
}

std::vector<std::string> GgufMetadata::strings(std::string_view key) const {
    const auto* ctx = impl_->metadata.get();
    const auto id = key_id(ctx, key);
    require(
        gguf_get_kv_type(ctx, id) == GGUF_TYPE_ARRAY
            && gguf_get_arr_type(ctx, id) == GGUF_TYPE_STRING,
        "gguf_metadata_strings"
    );
    const auto n = gguf_get_arr_n(ctx, id);
    require(n <= 524288, "gguf_array_limit");
    std::vector<std::string> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        out.emplace_back(gguf_get_arr_str(ctx, id, i));
    return out;
}

std::vector<std::int64_t> GgufMetadata::integers(std::string_view key) const {
    const auto* ctx = impl_->metadata.get();
    const auto id = key_id(ctx, key);
    require(
        gguf_get_kv_type(ctx, id) == GGUF_TYPE_ARRAY
            && gguf_get_arr_type(ctx, id) == GGUF_TYPE_INT32,
        "gguf_metadata_integers"
    );
    const auto n = gguf_get_arr_n(ctx, id);
    require(n <= 524288, "gguf_array_limit");
    const auto* raw = static_cast<const std::int32_t*>(gguf_get_arr_data(ctx, id));
    return {raw, raw + n};
}
}  // namespace kadan::checkpoint
