#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

// A minimal RFC 6455 client: the worker's only transport to the API. The
// Docker image's libcurl predates WebSocket support, so framing lives here and
// TLS uses the OpenSSL the worker already links.
namespace kadan::serving::websocket {
struct Url {
    bool secure = false;
    std::string host, port, path;

    static Url parse(std::string_view text);
};

enum class Opcode : std::uint8_t {
    continuation = 0,
    text = 1,
    binary = 2,
    close = 8,
    ping = 9,
    pong = 10
};

struct Frame {
    bool fin;
    Opcode opcode;
    std::string payload;
};

// Client frames are always masked; server frames never are.
std::string
encode_frame(Opcode opcode, std::string_view payload, std::array<std::uint8_t, 4> mask);

// Decodes unmasked server frames from a byte stream.
class FrameReader {
public:
    explicit FrameReader(std::size_t maximum) : maximum_(maximum) {
    }

    void feed(std::string_view bytes);
    std::optional<Frame> next();

private:
    std::string buffer_;
    std::size_t maximum_;
};

// Joins fragments and answers control frames; yields whole text messages.
class MessageAssembler {
public:
    explicit MessageAssembler(std::size_t maximum) : maximum_(maximum) {
    }

    struct Result {
        std::optional<std::string> message, pong;
        bool closed = false;
    };

    Result add(Frame frame);

private:
    std::string partial_;
    bool assembling_ = false;
    std::size_t maximum_;
};

std::string accept_key(std::string_view key);
std::string
handshake_request(const Url& url, std::string_view key, std::string_view token);
// Validates the 101 response headers; throws on anything else.
void check_handshake_response(std::string_view response, std::string_view key);

class Client {
public:
    Client(const Url& url, const std::string& token, std::size_t maximum);
    ~Client();
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    // Safe from any thread.
    void send_text(std::string_view message);
    // One reader thread. Returns nothing when the wait elapses without a whole
    // message; throws once the connection is closed or fails.
    std::optional<std::string> receive(std::chrono::milliseconds wait);
    // Asks the server for a pong; any frame received counts as hearing from it.
    void ping();

    std::chrono::steady_clock::time_point last_heard() const {
        return last_heard_;
    }

    void close() noexcept;

private:
    void write_all(std::string_view bytes);
    std::size_t read_some(char* buffer, std::size_t size);
    bool wait_readable(std::chrono::milliseconds wait);

    int socket_ = -1;
    void* context_ = nullptr;
    void* tls_ = nullptr;
    std::mutex io_;
    FrameReader reader_;
    MessageAssembler assembler_;
    std::chrono::steady_clock::time_point last_heard_ =
        std::chrono::steady_clock::now();
};
}  // namespace kadan::serving::websocket
