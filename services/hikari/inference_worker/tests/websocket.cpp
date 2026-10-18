#include "kadan/websocket.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace kadan::serving::websocket;

namespace {
// Server frames are unmasked; build them the way the API would.
std::string server_frame(Opcode opcode, std::string_view payload, bool fin = true) {
    std::string frame;
    frame.push_back(char((fin ? 0x80 : 0) | std::uint8_t(opcode)));
    if (payload.size() < 126) {
        frame.push_back(char(payload.size()));
    } else if (payload.size() <= 0xffff) {
        frame.push_back(char(126));
        frame.push_back(char(payload.size() >> 8));
        frame.push_back(char(payload.size()));
    } else {
        frame.push_back(char(127));
        for (int shift = 56; shift >= 0; shift -= 8)
            frame.push_back(char(std::uint64_t(payload.size()) >> shift));
    }
    frame.append(payload);
    return frame;
}

template <class Function> bool throws(Function function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void urls_parse_schemes_ports_and_paths() {
    const auto plain = Url::parse("ws://127.0.0.1:8000/v1/workers");
    assert(!plain.secure && plain.host == "127.0.0.1" && plain.port == "8000");
    assert(plain.path == "/v1/workers");
    const auto secure = Url::parse("wss://api.example.com");
    assert(secure.secure && secure.port == "443" && secure.path == "/");
    const auto ipv6 = Url::parse("ws://[::1]:9000/x");
    assert(ipv6.host == "::1" && ipv6.port == "9000");
    assert(throws([] { Url::parse("http://host/"); }));
    assert(throws([] { Url::parse("ws://host:http/"); }));
    assert(throws([] { Url::parse("ws://:80/"); }));
}

void client_frames_are_masked_with_sized_headers() {
    const std::array<std::uint8_t, 4> mask{1, 2, 3, 4};
    const auto small = encode_frame(Opcode::text, "hi", mask);
    assert(std::uint8_t(small[0]) == 0x81 && std::uint8_t(small[1]) == (0x80 | 2));
    assert(small.substr(2, 4) == std::string("\x01\x02\x03\x04", 4));
    assert(small[6] == char('h' ^ 1) && small[7] == char('i' ^ 2));
    const auto medium = encode_frame(Opcode::text, std::string(300, 'a'), mask);
    assert(std::uint8_t(medium[1]) == (0x80 | 126) && medium.size() == 4 + 4 + 300);
    const auto large = encode_frame(Opcode::binary, std::string(70000, 'a'), mask);
    assert(std::uint8_t(large[1]) == (0x80 | 127) && large.size() == 10 + 4 + 70000);
}

void reader_waits_for_whole_frames() {
    FrameReader reader(1 << 20);
    const auto frame = server_frame(Opcode::text, std::string(300, 'x'));
    reader.feed(std::string_view(frame).substr(0, 3));
    assert(!reader.next());
    reader.feed(std::string_view(frame).substr(3, 100));
    assert(!reader.next());
    reader.feed(std::string_view(frame).substr(103));
    const auto parsed = reader.next();
    assert(parsed && parsed->fin && parsed->opcode == Opcode::text);
    assert(parsed->payload == std::string(300, 'x') && !reader.next());
}

void reader_rejects_masked_oversized_and_fragmented_control_frames() {
    FrameReader masked(1 << 20);
    masked.feed(encode_frame(Opcode::text, "x", {1, 2, 3, 4}));
    assert(throws([&] { masked.next(); }));
    FrameReader bounded(10);
    bounded.feed(server_frame(Opcode::text, std::string(11, 'x')));
    assert(throws([&] { bounded.next(); }));
    FrameReader control(1 << 20);
    control.feed(server_frame(Opcode::ping, "", false));
    assert(throws([&] { control.next(); }));
}

void assembler_joins_fragments_and_answers_control_frames() {
    MessageAssembler assembler(1 << 20);
    auto result = assembler.add({false, Opcode::text, "hel"});
    assert(!result.message);
    result = assembler.add({true, Opcode::ping, "beat"});
    assert(result.pong == "beat" && !result.message);
    result = assembler.add({true, Opcode::continuation, "lo"});
    assert(result.message == "hello");
    assert(assembler.add({true, Opcode::close, ""}).closed);
    assert(throws([&] { assembler.add({true, Opcode::continuation, "x"}); }));
    MessageAssembler bounded(4);
    bounded.add({false, Opcode::text, "abc"});
    assert(throws([&] { bounded.add({true, Opcode::continuation, "de"}); }));
}

void handshake_carries_the_token_and_checks_accept() {
    // RFC 6455 section 1.3 example.
    assert(accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    const auto request = handshake_request(
        Url::parse("ws://localhost:8000/v1/workers"),
        "dGhlIHNhbXBsZSBub25jZQ==",
        "secret"
    );
    assert(request.starts_with("GET /v1/workers HTTP/1.1\r\n"));
    assert(request.find("Host: localhost:8000\r\n") != std::string::npos);
    assert(request.find("Authorization: Bearer secret\r\n") != std::string::npos);
    assert(request.ends_with("\r\n\r\n"));
    check_handshake_response(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
        "sec-websocket-accept:  s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n",
        "dGhlIHNhbXBsZSBub25jZQ=="
    );
    assert(throws([] {
        check_handshake_response(
            "HTTP/1.1 101 Switching Protocols\r\nSec-WebSocket-Accept: wrong\r\n",
            "dGhlIHNhbXBsZSBub25jZQ=="
        );
    }));
    bool rejected = false;
    try {
        check_handshake_response("HTTP/1.1 403 Forbidden\r\n", "x");
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()) == "worker_token_rejected";
    }
    assert(rejected);
}
}  // namespace

int main() {
    urls_parse_schemes_ports_and_paths();
    client_frames_are_masked_with_sized_headers();
    reader_waits_for_whole_frames();
    reader_rejects_masked_oversized_and_fragmented_control_frames();
    assembler_joins_fragments_and_answers_control_frames();
    handshake_carries_the_token_and_checks_accept();
    std::cout << "WebSocket framing units passed\n";
}
