#pragma once
#include <string_view>
namespace kadan::video::detail {
// Bounded parser for the released ComfyUI marker schema and key order. Spaces
// are accepted only between complete JSON tokens, never inside strings/numbers.
// Reject unknown/duplicate fields, escaped spellings, and trailing bytes.
inline unsigned convrot_group(std::string_view input) {
    if(input.size()>256)return 0;
    auto ws=[&]{while(!input.empty() && (input.front()==' ' || input.front()=='\n' || input.front()=='\r' || input.front()=='\t'))input.remove_prefix(1);};
    auto take=[&](std::string_view token){ws();if(!input.starts_with(token))return false;input.remove_prefix(token.size());return true;};
    if(!take("{") || !take("\"format\"") || !take(":") || !take("\"int8_tensorwise\"") || !take(",") || !take("\"convrot\"") || !take(":") || !take("true") || !take(",") || !take("\"convrot_groupsize\"") || !take(":"))return 0;
    unsigned group=0;if(take("256"))group=256;else if(take("64"))group=64;else return 0;
    if(!take("}"))return 0;
    ws();return input.empty()?group:0;
}
}
