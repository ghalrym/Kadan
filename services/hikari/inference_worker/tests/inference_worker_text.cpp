#include <cassert>
#include <iostream>

#include "kadan/utf8_stream.hpp"

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
    std::cout << "UTF-8 streaming passed; no inference\n";
}
