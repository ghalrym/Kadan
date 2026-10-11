#pragma once
#include "kadan/h3_profile.hpp"
#include "kadan/resources.hpp"
#include <algorithm>

namespace kadan::video {
// Bound the largest streamed execution stage, not the entire checkpoint.
inline Bytes h3_request_memory(const H3Profile &profile,
                               std::size_t prompt_bytes) {
  constexpr Bytes gib = 1024ULL * 1024 * 1024;
  const Bytes text = std::min<Bytes>(128000, 4 * prompt_bytes);
  const Bytes video = profile.video_tokens, audio = profile.audio_tokens;
  const Bytes height = profile.height / 16, width = profile.width / 16;
  const Bytes caller =
      32 * 1024 * 1024 +
      (3 * video * 96 + 2 * audio * 32 + 7 * height * width * 24) * 4 +
      profile.width * profile.height * 399 + (text + video + audio) * 20 +
      text * 5120 * 4;
  const Bytes patches = 7 * height * width, tokens = patches + 5;
  const Bytes decoder = 4 * (tokens * (4099 + 14336 + 16384) + patches * 3072);
  const Bytes execution = std::max(
      {496128 * (text + video + audio), decoder, Bytes(512 * 1024 * 1024)});
  return ((execution + caller + 4 * gib + gib - 1) / gib) * gib;
}
} // namespace kadan::video
