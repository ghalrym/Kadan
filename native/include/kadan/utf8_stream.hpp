#pragma once
#include <string>

namespace kadan::serving {
// Byte-level token decoding uses replacement for malformed or incomplete final
// sequences, while ordinary token boundaries retain their unfinished suffix.
inline std::string utf8_chunk(std::string &pending, bool final = false) {
  std::string output;
  std::size_t offset = 0;
  while (offset < pending.size()) {
    const auto first = static_cast<unsigned char>(pending[offset]);
    const unsigned width = first < 0x80                     ? 1
                           : first >= 0xc2 && first <= 0xdf ? 2
                           : first >= 0xe0 && first <= 0xef ? 3
                           : first >= 0xf0 && first <= 0xf4 ? 4
                                                            : 0;
    if (width && offset + width > pending.size()) {
      if (final) {
        output += "\xef\xbf\xbd";
        offset = pending.size();
      }
      break;
    }
    bool valid = width != 0;
    for (unsigned index = 1; index < width; ++index) {
      const auto byte = static_cast<unsigned char>(pending[offset + index]);
      valid = valid && byte >= 0x80 && byte <= 0xbf;
      if (index == 1)
        valid = valid && !(first == 0xe0 && byte < 0xa0) &&
                !(first == 0xed && byte >= 0xa0) &&
                !(first == 0xf0 && byte < 0x90) &&
                !(first == 0xf4 && byte >= 0x90);
    }
    if (valid) {
      output.append(pending, offset, width);
      offset += width;
    } else {
      output += "\xef\xbf\xbd";
      ++offset;
    }
  }
  pending.erase(0, offset);
  return output;
}
} // namespace kadan::serving
