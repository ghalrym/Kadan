#include "kadan/utf8_stream.hpp"
#include "kadan/h3_memory.hpp"
#include <cassert>
#include <iostream>

int main() {
    using kadan::serving::utf8_chunk;
    std::string pending = "hello \xe2";
    assert(utf8_chunk(pending) == "hello ");
    assert(pending == "\xe2");
    pending += "\x82\xac";
    assert(utf8_chunk(pending) == "\xe2\x82\xac" && pending.empty());
    pending = "\xf0\x9f";
    assert(utf8_chunk(pending, true) == "\xef\xbf\xbd" && pending.empty());
    pending = "\xff";
    assert(utf8_chunk(pending) == "\xef\xbf\xbd" && pending.empty());
    const auto small = kadan::video::h3_api_profile(480, "1:1", 4);
    const auto large = kadan::video::h3_api_profile(768, "16:9", 8);
    assert(kadan::video::h3_request_memory(large, 100) > kadan::video::h3_request_memory(small, 100));
    assert(kadan::video::h3_request_memory(small, 8000) > kadan::video::h3_request_memory(small, 100));
    std::cout << "UTF-8 streaming and H3 preparation arithmetic passed; no inference\n";
}
