#include <array>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <openssl/evp.h>
#include <unistd.h>

#include "kadan/preparation.hpp"

namespace kadan::serving {
void preparation_check(bool condition, const std::string& message, int status) {
    if (!condition)
        throw RequestError(message, status);
}

void preparation_cancel(const std::atomic_bool& cancel) {
    preparation_check(!cancel.load(), "Inference cancelled.", 499);
}

std::string read_bytes(const fs::path& path, std::size_t maximum) {
    preparation_check(
        !fs::is_symlink(path) && fs::is_regular_file(path),
        "Expected a regular file: " + path.string()
    );
    const auto size = fs::file_size(path);
    preparation_check(size <= maximum, "File exceeds size limit: " + path.string());
    std::ifstream input(path, std::ios::binary);
    std::string bytes(size, '\0');
    input.read(bytes.data(), static_cast<std::streamsize>(size));
    preparation_check(bool(input), "File read failed: " + path.string());
    return bytes;
}

Json read_document(const fs::path& path, std::size_t maximum) {
    return Json::parse(read_bytes(path, maximum));
}

void write_document(const fs::path& path, const Json& value) {
    std::ofstream output(path, std::ios::binary);
    output << value.dump();
    output.close();
    preparation_check(!output.fail(), "File write failed: " + path.string());
}

std::string base64_encode(std::string_view bytes) {
    preparation_check(bytes.size() <= 128 * 1024 * 1024, "Base64 input limit");
    std::string encoded(4 * ((bytes.size() + 2) / 3) + 1, '\0');
    const auto size = EVP_EncodeBlock(
        reinterpret_cast<unsigned char*>(encoded.data()),
        reinterpret_cast<const unsigned char*>(bytes.data()),
        static_cast<int>(bytes.size())
    );
    preparation_check(size >= 0, "Base64 encoding failed");
    encoded.resize(size);
    return encoded;
}

std::string base64_decode(std::string_view encoded, std::size_t maximum) {
    preparation_check(
        encoded.size() <= maximum && encoded.size() <= 128 * 1024 * 1024
            && encoded.size() % 4 == 0,
        "Invalid base64 length",
        422
    );
    if (encoded.empty())
        return {};
    auto end = encoded.find('=');
    if (end == std::string_view::npos)
        end = encoded.size();
    preparation_check(encoded.size() - end <= 2, "Invalid base64 padding", 422);
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        const char c = encoded[i];
        preparation_check(
            i >= end ? c == '='
                     : ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
                        || (c >= '0' && c <= '9') || c == '+' || c == '/'),
            "Invalid base64 character",
            422
        );
    }
    std::string bytes(encoded.size() / 4 * 3, '\0');
    const auto size = EVP_DecodeBlock(
        reinterpret_cast<unsigned char*>(bytes.data()),
        reinterpret_cast<const unsigned char*>(encoded.data()),
        static_cast<int>(encoded.size())
    );
    preparation_check(size >= 0, "Invalid base64 data", 422);
    bytes.resize(size - (encoded.size() - end));
    return bytes;
}

std::string file_digest(const fs::path& path, const std::atomic_bool& cancel) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest(
        EVP_MD_CTX_new(), EVP_MD_CTX_free
    );
    preparation_check(
        digest && EVP_DigestInit_ex(digest.get(), EVP_sha256(), nullptr) == 1,
        "Digest initialization failed"
    );
    std::ifstream input(path, std::ios::binary);
    preparation_check(bool(input), "Cannot hash checkpoint");
    std::array<char, 1024 * 1024> buffer{};
    while (input) {
        preparation_cancel(cancel);
        input.read(buffer.data(), buffer.size());
        preparation_check(
            EVP_DigestUpdate(digest.get(), buffer.data(), input.gcount()) == 1,
            "Digest update failed"
        );
    }
    preparation_check(input.eof(), "Checkpoint hash read failed");
    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
    unsigned count = 0;
    preparation_check(
        EVP_DigestFinal_ex(digest.get(), bytes.data(), &count) == 1, "Digest failed"
    );
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned i = 0; i < count; ++i) {
        result += hex[bytes[i] >> 4];
        result += hex[bytes[i] & 15];
    }
    return result;
}

fs::path worker_data_directory() {
    if (const auto* configured = std::getenv("KADAN_WORKER_DATA"))
        return configured;
    std::array<char, 4096> executable{};
    const auto size = readlink("/proc/self/exe", executable.data(), executable.size());
    if (size > 0 && std::size_t(size) < executable.size()) {
        auto installed =
            fs::path(std::string(executable.data(), size)).parent_path().parent_path()
            / "share/kadan";
        if (fs::is_regular_file(installed / "catalog.json"))
            return installed;
    }
    return KADAN_WORKER_SOURCE_DATA;
}
}  // namespace kadan::serving
