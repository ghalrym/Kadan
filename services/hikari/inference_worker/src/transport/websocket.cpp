#include "kadan/websocket.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <openssl/ssl.h>
#include <poll.h>
#include <random>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace kadan::serving::websocket {
namespace {
void need(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

std::string base64(const unsigned char* data, std::size_t size) {
    std::string encoded(4 * ((size + 2) / 3), '\0');
    const auto written = EVP_EncodeBlock(
        reinterpret_cast<unsigned char*>(encoded.data()), data, int(size)
    );
    encoded.resize(std::size_t(written));
    return encoded;
}

std::string lower(std::string_view text) {
    std::string result(text);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return char(std::tolower(c));
    });
    return result;
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t'))
        text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t'))
        text.remove_suffix(1);
    return text;
}

std::array<std::uint8_t, 4> random_mask() {
    static thread_local std::random_device device;
    const auto value = device();
    return {
        std::uint8_t(value),
        std::uint8_t(value >> 8),
        std::uint8_t(value >> 16),
        std::uint8_t(value >> 24)
    };
}

std::string random_key() {
    static thread_local std::random_device device;
    std::array<unsigned char, 16> bytes;
    for (auto& byte : bytes)
        byte = static_cast<unsigned char>(device());
    return base64(bytes.data(), bytes.size());
}

SSL* tls(void* pointer) {
    return static_cast<SSL*>(pointer);
}

constexpr auto io_timeout = std::chrono::seconds(30);
}  // namespace

Url Url::parse(std::string_view text) {
    Url url;
    if (text.starts_with("ws://")) {
        text.remove_prefix(5);
    } else if (text.starts_with("wss://")) {
        url.secure = true;
        text.remove_prefix(6);
    } else
        throw std::invalid_argument("websocket_url_scheme");
    const auto slash = text.find('/');
    auto authority = text.substr(0, slash);
    url.path = slash == std::string_view::npos ? "/" : std::string(text.substr(slash));
    if (authority.starts_with('[')) {
        const auto close = authority.find(']');
        need(close != std::string_view::npos, "websocket_url_host");
        url.host = std::string(authority.substr(1, close - 1));
        authority.remove_prefix(close + 1);
        if (authority.starts_with(':'))
            url.port = std::string(authority.substr(1));
        else
            need(authority.empty(), "websocket_url_host");
    } else {
        const auto colon = authority.rfind(':');
        url.host = std::string(authority.substr(0, colon));
        if (colon != std::string_view::npos)
            url.port = std::string(authority.substr(colon + 1));
    }
    if (url.port.empty())
        url.port = url.secure ? "443" : "80";
    need(!url.host.empty(), "websocket_url_host");
    need(
        url.port.find_first_not_of("0123456789") == std::string::npos
            && url.port.size() <= 5,
        "websocket_url_port"
    );
    return url;
}

std::string encode_frame(
    Opcode opcode, std::string_view payload, std::array<std::uint8_t, 4> mask
) {
    std::string frame;
    frame.reserve(payload.size() + 14);
    frame.push_back(char(0x80 | std::uint8_t(opcode)));
    const auto size = payload.size();
    if (size < 126) {
        frame.push_back(char(0x80 | size));
    } else if (size <= 0xffff) {
        frame.push_back(char(0x80 | 126));
        frame.push_back(char(size >> 8));
        frame.push_back(char(size));
    } else {
        frame.push_back(char(0x80 | 127));
        for (int shift = 56; shift >= 0; shift -= 8)
            frame.push_back(char(std::uint64_t(size) >> shift));
    }
    for (auto byte : mask)
        frame.push_back(char(byte));
    const auto start = frame.size();
    frame.append(payload);
    for (std::size_t i = 0; i < size; ++i)
        frame[start + i] = char(std::uint8_t(frame[start + i]) ^ mask[i % 4]);
    return frame;
}

void FrameReader::feed(std::string_view bytes) {
    buffer_.append(bytes);
}

std::optional<Frame> FrameReader::next() {
    if (buffer_.size() < 2)
        return std::nullopt;
    const auto first = std::uint8_t(buffer_[0]), second = std::uint8_t(buffer_[1]);
    need((first & 0x70) == 0, "websocket_reserved_bits");
    need((second & 0x80) == 0, "websocket_masked_server_frame");
    std::size_t header = 2;
    std::uint64_t size = second & 0x7f;
    if (size == 126) {
        if (buffer_.size() < 4)
            return std::nullopt;
        size =
            (std::uint64_t(std::uint8_t(buffer_[2])) << 8) | std::uint8_t(buffer_[3]);
        header = 4;
    } else if (size == 127) {
        if (buffer_.size() < 10)
            return std::nullopt;
        size = 0;
        for (std::size_t i = 2; i < 10; ++i)
            size = (size << 8) | std::uint8_t(buffer_[i]);
        header = 10;
    }
    need(size <= maximum_, "websocket_frame_bound");
    const auto opcode = Opcode(first & 0x0f);
    const bool control = std::uint8_t(opcode) >= 8;
    need(!control || (size <= 125 && (first & 0x80)), "websocket_control_frame");
    if (buffer_.size() < header + size)
        return std::nullopt;
    Frame frame{bool(first & 0x80), opcode, buffer_.substr(header, size)};
    buffer_.erase(0, header + size);
    return frame;
}

MessageAssembler::Result MessageAssembler::add(Frame frame) {
    Result result;
    switch (frame.opcode) {
    case Opcode::ping:
        result.pong = std::move(frame.payload);
        return result;
    case Opcode::pong:
        return result;
    case Opcode::close:
        result.closed = true;
        return result;
    case Opcode::text:
    case Opcode::binary:
        need(!assembling_, "websocket_interleaved_message");
        partial_ = std::move(frame.payload);
        assembling_ = true;
        break;
    case Opcode::continuation:
        need(assembling_, "websocket_orphan_continuation");
        need(
            partial_.size() + frame.payload.size() <= maximum_,
            "websocket_message_bound"
        );
        partial_ += frame.payload;
        break;
    default:
        throw std::runtime_error("websocket_opcode");
    }
    if (frame.fin) {
        assembling_ = false;
        result.message = std::move(partial_);
        partial_.clear();
    }
    return result;
}

std::string accept_key(std::string_view key) {
    const std::string source =
        std::string(key) + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::array<unsigned char, SHA_DIGEST_LENGTH> digest;
    SHA1(
        reinterpret_cast<const unsigned char*>(source.data()),
        source.size(),
        digest.data()
    );
    return base64(digest.data(), digest.size());
}

std::string
handshake_request(const Url& url, std::string_view key, std::string_view token) {
    const bool default_port = url.port == (url.secure ? "443" : "80");
    const auto host =
        url.host.find(':') == std::string::npos ? url.host : "[" + url.host + "]";
    std::string request = "GET " + url.path + " HTTP/1.1\r\n";
    request += "Host: " + host + (default_port ? "" : ":" + url.port) + "\r\n";
    request += "Upgrade: websocket\r\nConnection: Upgrade\r\n";
    request += "Sec-WebSocket-Key: " + std::string(key) + "\r\n";
    request += "Sec-WebSocket-Version: 13\r\n";
    request += "Authorization: Bearer " + std::string(token) + "\r\n\r\n";
    return request;
}

void check_handshake_response(std::string_view response, std::string_view key) {
    const auto line_end = response.find("\r\n");
    need(line_end != std::string_view::npos, "websocket_handshake");
    const auto status = response.substr(0, line_end);
    if (status.find(" 401") != std::string_view::npos
        || status.find(" 403") != std::string_view::npos)
        throw std::runtime_error("worker_token_rejected");
    need(status.starts_with("HTTP/1.1 101"), "websocket_handshake_status");
    bool accepted = false;
    auto rest = response.substr(line_end + 2);
    while (!rest.empty()) {
        const auto end = rest.find("\r\n");
        const auto line = rest.substr(0, end);
        rest =
            end == std::string_view::npos ? std::string_view{} : rest.substr(end + 2);
        const auto colon = line.find(':');
        if (colon == std::string_view::npos)
            continue;
        if (lower(trim(line.substr(0, colon))) == "sec-websocket-accept")
            accepted = trim(line.substr(colon + 1)) == accept_key(key);
    }
    need(accepted, "websocket_handshake_accept");
}

Client::Client(const Url& url, const std::string& token, std::size_t maximum)
    : reader_(maximum), assembler_(maximum) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    need(
        getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &addresses) == 0,
        "websocket_resolve"
    );
    for (auto* address = addresses; address; address = address->ai_next) {
        socket_ = ::socket(
            address->ai_family,
            address->ai_socktype | SOCK_CLOEXEC,
            address->ai_protocol
        );
        if (socket_ < 0)
            continue;
        if (::connect(socket_, address->ai_addr, address->ai_addrlen) == 0)
            break;
        ::close(socket_);
        socket_ = -1;
    }
    freeaddrinfo(addresses);
    need(socket_ >= 0, "websocket_connect");
    try {
        const int enabled = 1;
        setsockopt(socket_, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
        if (url.secure) {
            auto* context = SSL_CTX_new(TLS_client_method());
            need(context, "websocket_tls_context");
            context_ = context;
            SSL_CTX_set_verify(context, SSL_VERIFY_PEER, nullptr);
            need(SSL_CTX_set_default_verify_paths(context) == 1, "websocket_tls_roots");
            auto* session = SSL_new(context);
            need(session, "websocket_tls_session");
            tls_ = session;
            need(
                SSL_set_tlsext_host_name(session, url.host.c_str()) == 1
                    && SSL_set1_host(session, url.host.c_str()) == 1
                    && SSL_set_fd(session, socket_) == 1,
                "websocket_tls_host"
            );
            need(SSL_connect(session) == 1, "websocket_tls_handshake");
        }
        const auto key = random_key();
        const auto request = handshake_request(url, key, token);
        std::size_t sent = 0;
        while (sent < request.size()) {
            const auto count =
                tls_ ? SSL_write(
                           tls(tls_), request.data() + sent, int(request.size() - sent)
                       )
                     : ::send(
                           socket_,
                           request.data() + sent,
                           request.size() - sent,
                           MSG_NOSIGNAL
                       );
            need(count > 0, "websocket_handshake_write");
            sent += std::size_t(count);
        }
        std::string response;
        std::array<char, 4096> buffer;
        std::size_t header_end = std::string::npos;
        while (header_end == std::string::npos) {
            need(response.size() < 64 * 1024, "websocket_handshake_bound");
            const auto count =
                tls_ ? SSL_read(tls(tls_), buffer.data(), int(buffer.size()))
                     : ::recv(socket_, buffer.data(), buffer.size(), 0);
            need(count > 0, "websocket_handshake_read");
            response.append(buffer.data(), std::size_t(count));
            header_end = response.find("\r\n\r\n");
        }
        check_handshake_response(response.substr(0, header_end + 2), key);
        reader_.feed(std::string_view(response).substr(header_end + 4));
        const auto flags = fcntl(socket_, F_GETFL);
        need(
            flags >= 0 && fcntl(socket_, F_SETFL, flags | O_NONBLOCK) == 0,
            "websocket_nonblocking"
        );
    } catch (...) {
        close();
        throw;
    }
}

Client::~Client() {
    close();
}

void Client::close() noexcept {
    std::lock_guard guard(io_);
    if (tls_) {
        SSL_free(tls(tls_));
        tls_ = nullptr;
    }
    if (context_) {
        SSL_CTX_free(static_cast<SSL_CTX*>(context_));
        context_ = nullptr;
    }
    if (socket_ >= 0) {
        ::shutdown(socket_, SHUT_RDWR);
        ::close(socket_);
        socket_ = -1;
    }
}

void Client::send_text(std::string_view message) {
    write_all(encode_frame(Opcode::text, message, random_mask()));
}

void Client::ping() {
    write_all(encode_frame(Opcode::ping, "", random_mask()));
}

void Client::write_all(std::string_view bytes) {
    std::lock_guard guard(io_);
    need(socket_ >= 0, "websocket_closed");
    const auto deadline = std::chrono::steady_clock::now() + io_timeout;
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        need(std::chrono::steady_clock::now() < deadline, "websocket_write_timeout");
        const auto remaining = bytes.size() - offset;
        bool wait_read = false;
        if (tls_) {
            const auto count = SSL_write(
                tls(tls_),
                bytes.data() + offset,
                int(std::min<std::size_t>(remaining, 1 << 30))
            );
            if (count > 0) {
                offset += std::size_t(count);
                continue;
            }
            const auto error = SSL_get_error(tls(tls_), count);
            need(
                error == SSL_ERROR_WANT_WRITE || error == SSL_ERROR_WANT_READ,
                "websocket_write"
            );
            wait_read = error == SSL_ERROR_WANT_READ;
        } else {
            const auto count =
                ::send(socket_, bytes.data() + offset, remaining, MSG_NOSIGNAL);
            if (count > 0) {
                offset += std::size_t(count);
                continue;
            }
            need(count < 0 && (errno == EAGAIN || errno == EINTR), "websocket_write");
        }
        pollfd descriptor{socket_, short(wait_read ? POLLIN : POLLOUT), 0};
        const auto ready = poll(&descriptor, 1, 100);
        need(ready >= 0 || errno == EINTR, "websocket_write_poll");
        need(
            ready <= 0 || !(descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)),
            "websocket_closed"
        );
    }
}

bool Client::wait_readable(std::chrono::milliseconds wait) {
    {
        std::lock_guard guard(io_);
        need(socket_ >= 0, "websocket_closed");
        if (tls_ && SSL_pending(tls(tls_)) > 0)
            return true;
    }
    pollfd descriptor{socket_, POLLIN, 0};
    const auto ready = poll(&descriptor, 1, int(wait.count()));
    if (ready < 0 && errno == EINTR)
        return false;
    need(ready >= 0, "websocket_read_poll");
    return ready > 0;
}

std::size_t Client::read_some(char* buffer, std::size_t size) {
    int socket = -1;
    {
        std::lock_guard guard(io_);
        need(socket_ >= 0, "websocket_closed");
        // One TLS session cannot read and write at once; a plain socket can.
        if (tls_) {
            const auto count = SSL_read(tls(tls_), buffer, int(size));
            if (count > 0)
                return std::size_t(count);
            const auto error = SSL_get_error(tls(tls_), count);
            need(
                error == SSL_ERROR_WANT_READ || error == SSL_ERROR_WANT_WRITE,
                "websocket_closed"
            );
            return 0;
        }
        socket = socket_;
    }
    // Non-blocking: wait_readable() polls first, and only this thread closes.
    const auto count = ::recv(socket, buffer, size, 0);
    if (count > 0)
        return std::size_t(count);
    need(count < 0 && (errno == EAGAIN || errno == EINTR), "websocket_closed");
    return 0;
}

std::optional<std::string> Client::receive(std::chrono::milliseconds wait) {
    const auto deadline = std::chrono::steady_clock::now() + wait;
    std::vector<char> buffer(256 * 1024);
    for (;;) {
        while (auto frame = reader_.next()) {
            last_heard_ = std::chrono::steady_clock::now();
            auto result = assembler_.add(std::move(*frame));
            if (result.pong)
                write_all(encode_frame(Opcode::pong, *result.pong, random_mask()));
            if (result.closed) {
                try {
                    write_all(encode_frame(Opcode::close, "", random_mask()));
                } catch (const std::exception&) {
                }
                throw std::runtime_error("websocket_closed");
            }
            if (result.message)
                return result.message;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline)
            return std::nullopt;
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        if (!wait_readable(std::max(remaining, std::chrono::milliseconds(1))))
            continue;
        const auto count = read_some(buffer.data(), buffer.size());
        reader_.feed(std::string_view(buffer.data(), count));
    }
}
}  // namespace kadan::serving::websocket
