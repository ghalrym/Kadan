#include "kadan/checkpoint.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace kadan::checkpoint {
namespace {
void require(bool ok, const char* error) { if (!ok) throw std::invalid_argument(error); }
std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    require(b <= UINT64_MAX - a, "size_overflow"); return a + b;
}
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    require(b == 0 || a <= UINT64_MAX / b, "size_overflow"); return a * b;
}
class Fd {
public:
    explicit Fd(int fd) : value(fd) { if (fd < 0) throw std::runtime_error("open_failed"); }
    ~Fd() { if (value >= 0) ::close(value); }
    Fd(Fd&& other) noexcept : value(std::exchange(other.value, -1)) {}
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    int value;
};
struct stat regular_file(int fd) {
    struct stat s{};
    if (::fstat(fd, &s) != 0) throw std::runtime_error("stat_failed");
    require(S_ISREG(s.st_mode) && s.st_size >= 8, "not_safetensors_file");
    return s;
}
void read_at(int fd, std::uint64_t offset, std::span<std::uint8_t> bytes) {
    require(add(offset, bytes.size()) <= static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()), "file_offset_overflow");
    while (!bytes.empty()) {
        const auto chunk = std::min(bytes.size(), std::size_t{1024 * 1024});
        const auto n = ::pread(fd, bytes.data(), chunk, static_cast<off_t>(offset));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::runtime_error("short_or_failed_read");
        offset += static_cast<std::size_t>(n); bytes = bytes.subspan(static_cast<std::size_t>(n));
    }
}
std::uint64_t little64(const std::array<std::uint8_t, 8>& bytes) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t{bytes[i]} << (8 * i);
    return value;
}
void basename(std::string_view name) {
    require(!name.empty() && name.size() <= 255 && name != "." && name != "..", "invalid_shard_name");
    for (const unsigned char c : name)
        require((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-', "invalid_shard_name");
}
struct Tensor {
    explicit Tensor(std::pmr::memory_resource* resource) : name(resource) {}
    std::pmr::string name;
    Dtype dtype{};
    std::array<std::uint64_t, 8> shape{};
    std::size_t rank = 0;
    std::uint64_t begin = 0, end = 0;
    std::uint64_t elements() const {
        std::uint64_t n = 1;
        for (std::size_t i = 0; i < rank; ++i) n = mul(n, shape[i]);
        return n;
    }
};

// Narrow original parser for the safetensors schema, not a generic JSON DOM.
// ASCII strings (including ASCII escapes), unsigned decimal integers, depth
// fixed by the schema, and strict duplicate/unknown-field checks. Unsupported
// Unicode metadata fails closed; no recursive skip of arbitrary JSON is used.
class HeaderParser {
public:
    HeaderParser(std::string_view text, std::pmr::memory_resource* resource)
        : text_(text), resource_(resource) {}
    void parse(std::pmr::vector<Tensor>& tensors, std::size_t limit) {
        require(!text_.empty() && text_[0] == '{', "header_start");
        expect('{'); bool metadata = false;
        if (!take('}')) {
            do {
                auto name = string(); expect(':');
                if (name == "__metadata__") {
                    require(!metadata, "duplicate_metadata"); metadata = true; metadata_map();
                } else {
                    require(!name.empty() && tensors.size() < limit, "tensor_limit_or_name");
                    Tensor tensor(resource_); tensor.name = std::move(name); descriptor(tensor);
                    tensors.push_back(std::move(tensor));
                }
            } while (take(','));
            expect('}');
        }
        whitespace(); require(position_ == text_.size(), "trailing_json");
    }
private:
    void whitespace() {
        while (position_ < text_.size() && (text_[position_] == ' ' || text_[position_] == '\n' ||
               text_[position_] == '\r' || text_[position_] == '\t')) ++position_;
    }
    bool take(char c) {
        whitespace(); if (position_ == text_.size() || text_[position_] != c) return false;
        ++position_; return true;
    }
    void expect(char c) { require(take(c), "json_syntax"); }
    unsigned hex() {
        require(position_ < text_.size(), "json_escape"); const auto c = text_[position_++];
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::invalid_argument("json_escape");
    }
    std::pmr::string string() {
        expect('"'); std::pmr::string result(resource_);
        while (position_ < text_.size()) {
            unsigned char c = text_[position_++];
            if (c == '"') return result;
            require(c >= 32 && c < 128, "non_ascii_json_string");
            if (c == '\\') {
                require(position_ < text_.size(), "json_escape"); c = text_[position_++];
                switch (c) {
                    case '"': case '\\': case '/': break;
                    case 'b': c = 8; break; case 'f': c = 12; break;
                    case 'n': c = 10; break; case 'r': c = 13; break; case 't': c = 9; break;
                    case 'u': {
                        unsigned value = 0;
                        for (unsigned i = 0; i < 4; ++i) value = value * 16 + hex();
                        require(value != 0 && value < 128, "non_ascii_json_string"); c = value; break;
                    }
                    default: throw std::invalid_argument("json_escape");
                }
            }
            require(result.size() < 512, "json_string_limit"); result += static_cast<char>(c);
        }
        throw std::invalid_argument("unterminated_string");
    }
    std::uint64_t integer() {
        whitespace(); const auto first = position_;
        while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
        require(position_ > first && (position_ - first == 1 || text_[first] != '0'), "json_integer");
        std::uint64_t value = 0;
        const auto result = std::from_chars(text_.data() + first, text_.data() + position_, value);
        require(result.ec == std::errc{}, "json_integer"); return value;
    }
    void metadata_map() {
        std::pmr::vector<std::pmr::string> keys(resource_); expect('{');
        if (take('}')) return;
        do {
            auto key = string();
            require(keys.size() < 1024 && std::find(keys.begin(), keys.end(), key) == keys.end(), "metadata_key_limit_or_duplicate");
            keys.push_back(std::move(key)); expect(':'); string();
        } while (take(','));
        expect('}');
    }
    void descriptor(Tensor& tensor) {
        expect('{'); unsigned fields = 0;
        do {
            const auto key = string(); expect(':'); unsigned bit = 0;
            if (key == "dtype") {
                bit = 1; const auto dtype = string();
                if (dtype == "U8") tensor.dtype = Dtype::u8;
                else if (dtype == "F8_E4M3") tensor.dtype = Dtype::fp8;
                else if (dtype == "F32") tensor.dtype = Dtype::fp32;
                else if (dtype == "BF16") tensor.dtype = Dtype::bf16;
                else throw std::invalid_argument("unsupported_dtype");
            } else if (key == "shape") {
                bit = 2; expect('[');
                if (!take(']')) {
                    do {
                        require(tensor.rank < tensor.shape.size(), "rank_limit");
                        tensor.shape[tensor.rank++] = integer();
                    } while (take(',')); expect(']');
                }
            } else if (key == "data_offsets") {
                bit = 4; expect('['); tensor.begin = integer(); expect(','); tensor.end = integer(); expect(']');
            } else throw std::invalid_argument("unknown_tensor_field");
            require(!(fields & bit), "duplicate_tensor_field"); fields |= bit;
        } while (take(','));
        expect('}'); require(fields == 7, "missing_tensor_field");
        const auto width = tensor.dtype == Dtype::fp32 ? 4U : tensor.dtype == Dtype::bf16 ? 2U : 1U;
        require(tensor.end >= tensor.begin && tensor.end - tensor.begin == mul(tensor.elements(), width), "tensor_byte_size");
    }
    std::string_view text_;
    std::size_t position_ = 0;
    std::pmr::memory_resource* resource_;
};
} // namespace

std::size_t MemoryBudget::used() const { std::lock_guard lock(mutex_); return used_; }
void* MemoryBudget::do_allocate(std::size_t bytes, std::size_t alignment) {
    bytes = std::max(bytes, std::size_t{1});
    {
        std::lock_guard lock(mutex_);
        if (bytes > limit_ - used_) throw std::bad_alloc();
        used_ += bytes;
    }
    try {
        return alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__ ? ::operator new(bytes, std::align_val_t(alignment)) : ::operator new(bytes);
    } catch (...) { std::lock_guard lock(mutex_); used_ -= bytes; throw; }
}
void MemoryBudget::do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) {
    if (alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__) ::operator delete(pointer, std::align_val_t(alignment));
    else ::operator delete(pointer);
    std::lock_guard lock(mutex_); used_ -= std::max(bytes, std::size_t{1});
}
Projection::Projection(std::shared_ptr<MemoryBudget> budget)
    : budget_(std::move(budget)), weights_(budget_.get()), blocks_(budget_.get()), multipliers_(budget_.get()) {}
Projection::Projection(Projection&& other) noexcept
    : budget_(other.budget_), weights_(std::move(other.weights_)), blocks_(std::move(other.blocks_)),
      multipliers_(std::move(other.multipliers_)), encoding_(other.encoding_), rows_(other.rows_), columns_(other.columns_) {}
quantization::Matrix Projection::view() const & { return {encoding_, rows_, columns_, weights_, blocks_, multipliers_}; }

struct Shard::Impl {
    std::shared_ptr<MemoryBudget> budget;
    Fd file;
    struct stat identity;
    std::uint64_t data_begin = 0;
    std::pmr::vector<Tensor> tensors;
    Impl(Fd fd, std::shared_ptr<MemoryBudget> quota) : budget(std::move(quota)), file(std::move(fd)),
        identity(regular_file(file.value)), tensors(budget.get()) {}
    void unchanged() const {
        const auto s = regular_file(file.value);
        require(s.st_size == identity.st_size && s.st_mtim.tv_sec == identity.st_mtim.tv_sec &&
                s.st_mtim.tv_nsec == identity.st_mtim.tv_nsec && s.st_ctim.tv_sec == identity.st_ctim.tv_sec &&
                s.st_ctim.tv_nsec == identity.st_ctim.tv_nsec, "checkpoint_changed");
    }
    const Tensor& find(std::string_view name) const {
        const auto it = std::lower_bound(tensors.begin(), tensors.end(), name,
            [](const auto& tensor, auto key) { return std::string_view(tensor.name) < key; });
        require(it != tensors.end() && it->name == name, "missing_tensor"); return *it;
    }
    void read(const Tensor& tensor, std::uint64_t skip, std::span<std::uint8_t> destination) const {
        require(skip <= tensor.end - tensor.begin && destination.size() <= tensor.end - tensor.begin - skip, "tensor_read_bounds");
        read_at(file.value, add(data_begin, add(tensor.begin, skip)), destination);
    }
};
Shard::Shard(const char* root, std::string_view shard_name, std::shared_ptr<MemoryBudget> budget, Limits limits) {
    require(budget != nullptr && limits.header_bytes != 0 && limits.tensors != 0, "invalid_reader_limits");
    basename(shard_name);
    Fd directory(::open(root, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    std::array<char, 256> name{}; std::copy(shard_name.begin(), shard_name.end(), name.begin());
    // A local guard owns the descriptor before any allocation can throw.
    Fd source(::openat(directory.value, name.data(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC));
    // make_unique allocates before moving the guard into Impl; all failure paths
    // retain a descriptor owner, including allocation and metadata exceptions.
    impl_ = std::make_unique<Impl>(std::move(source), budget);
    std::array<std::uint8_t, 8> prefix{}; read_at(impl_->file.value, 0, prefix);
    const auto header_size = little64(prefix);
    require(header_size > 0 && header_size <= limits.header_bytes && header_size <= static_cast<std::uint64_t>(impl_->identity.st_size) - 8, "header_size");
    impl_->data_begin = add(8, header_size);
    std::pmr::vector<std::uint8_t> header(budget.get()); header.resize(header_size);
    read_at(impl_->file.value, 8, header);
    HeaderParser parser({reinterpret_cast<const char*>(header.data()), header.size()}, budget.get());
    parser.parse(impl_->tensors, limits.tensors);
    auto& tensors = impl_->tensors;
    std::sort(tensors.begin(), tensors.end(), [](const auto& a, const auto& b) {
        return a.begin == b.begin ? a.end < b.end : a.begin < b.begin;
    });
    std::uint64_t previous = 0;
    for (const auto& tensor : tensors) { require(tensor.begin == previous, "tensor_gap_or_overlap"); previous = tensor.end; }
    require(previous == static_cast<std::uint64_t>(impl_->identity.st_size) - impl_->data_begin, "payload_coverage");
    std::sort(tensors.begin(), tensors.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    for (std::size_t i = 1; i < tensors.size(); ++i) require(tensors[i - 1].name != tensors[i].name, "duplicate_tensor");
    impl_->unchanged();
}
Shard::~Shard() = default;
std::size_t Shard::tensor_count() const { return impl_->tensors.size(); }
TensorInfo Shard::tensor(std::string_view name) const {
    const auto& t = impl_->find(name);
    return {t.name, t.dtype, t.shape, t.rank, t.end - t.begin};
}
void Shard::check_unchanged() const { impl_->unchanged(); }
void Shard::read_tensor(std::string_view name, std::size_t offset, std::span<std::uint8_t> destination) const {
    const auto& t = impl_->find(name);
    impl_->unchanged(); impl_->read(t, offset, destination); impl_->unchanged();
}
Projection Shard::load_modelopt_rows(std::string_view prefix, std::size_t first,
                                    std::size_t count, std::size_t payload_budget, std::shared_ptr<MemoryBudget> payload_memory) const {
    require(!prefix.empty() && prefix.size() <= 480 && count != 0, "projection_range_or_name");
    auto name = [&](std::string_view suffix) { std::pmr::string n(prefix, impl_->budget.get()); n += suffix; return n; };
    const auto& weight = impl_->find(name(".weight"));
    const auto& scale = impl_->find(name(".weight_scale"));
    require(weight.rank == 2 && first <= weight.shape[0] && count <= weight.shape[0] - first && weight.shape[1] != 0, "projection_shape");
    require(weight.dtype == Dtype::u8 || weight.dtype == Dtype::fp8, "projection_dtype");
    const bool fp4 = weight.dtype == Dtype::u8;
    const auto columns = fp4 ? mul(weight.shape[1], 2) : weight.shape[1];
    const Tensor* global = nullptr;
    std::uint64_t scale_rows = 0, multiplier_count = 0;
    if (fp4) {
        require(columns % 16 == 0 && scale.dtype == Dtype::fp8 && scale.rank == 2 &&
                scale.shape[0] == weight.shape[0] && scale.shape[1] == columns / 16, "nvfp4_scale_layout");
        global = &impl_->find(name(".weight_scale_2"));
        require(global->dtype == Dtype::fp32 && (global->rank == 0 || (global->rank == 1 && global->shape[0] == 1)), "nvfp4_global_layout");
        scale_rows = mul(count, columns / 16); multiplier_count = 1;
    } else {
        const bool scalar = scale.rank == 0 || (scale.rank == 1 && scale.shape[0] == 1);
        const bool rows = (scale.rank == 1 || (scale.rank == 2 && scale.shape[1] == 1)) && scale.shape[0] == weight.shape[0];
        require(scale.dtype == Dtype::fp32 && (scalar || rows), "fp8_scale_layout");
        multiplier_count = scalar ? 1 : count;
    }
    const auto weight_bytes = mul(count, weight.shape[1]);
    const auto total = add(add(weight_bytes, scale_rows), mul(multiplier_count, sizeof(float)));
    require(total <= payload_budget && total <= SIZE_MAX && columns <= SIZE_MAX, "payload_budget");
    Projection result(payload_memory ? std::move(payload_memory) : impl_->budget);
    result.encoding_ = fp4 ? quantization::Encoding::modelopt_nvfp4 : quantization::Encoding::modelopt_fp8;
    result.rows_ = count; result.columns_ = columns;
    result.weights_.resize(weight_bytes); result.blocks_.resize(scale_rows); result.multipliers_.resize(multiplier_count);
    impl_->unchanged();
    impl_->read(weight, mul(first, weight.shape[1]), result.weights_);
    if (fp4) impl_->read(scale, mul(first, columns / 16), result.blocks_);
    const auto& multiplier = fp4 ? *global : scale;
    const bool scalar = multiplier.elements() == 1;
    for (std::size_t i = 0; i < multiplier_count; ++i) {
        std::array<std::uint8_t, 4> raw{};
        impl_->read(multiplier, scalar ? 0 : mul(add(first, i), 4), raw);
        result.multipliers_[i] = quantization::fp32_le(raw);
    }
    impl_->unchanged();
    // Zero-row decode runs #99's format/value validation without dense allocation.
    quantization::decode_rows(result.view(), 0, 0, 0);
    return result;
}
} // namespace kadan::checkpoint
