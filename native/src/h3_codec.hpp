#pragma once
#include <atomic>
#include <cstddef>
#include <string>
namespace kadan::video::h3 {
// Synchronous, bounded codec subprocess; caller owns/admitted input and output
// paths in a private directory and a 256 MiB codec memory envelope. Never shell
// interprets paths. On cancellation/failure the owned child is killed/reaped.
void encode_mp4(const std::string& input,const std::string& output,const std::atomic_bool& cancel,std::size_t expected_frames=0,const std::string& audio={});
}
