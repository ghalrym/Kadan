#pragma once
#include <cstddef>
#include <string_view>
namespace kadan::video {
struct H3Profile {
    std::size_t width,height,frames,video_time,audio_time,video_tokens,audio_tokens,vae_tokens,output_frames;
};
// The existing H3 API contract: 24 fps, 1–15 whole seconds, 480p/768p,
// portrait/landscape/square. Uses the checkpoint's 32-pixel canvas alignment,
// 768*1344 soft area limit and 17n+5 temporal alignment. These are model
// geometry rules; output_frames retains the requested duration.
// Memory admission remains a separate execution decision.
H3Profile h3_api_profile(std::size_t short_edge,std::string_view aspect,std::size_t seconds);
}
