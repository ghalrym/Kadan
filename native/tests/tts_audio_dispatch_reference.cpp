// Reuse diagnostic transport with an injected CPU backend for randomized parity.
#include "kadan/tts_audio.hpp"
#include "dense_reference.hpp"
namespace kadan::tts {
class DispatchAudioDecoder:public AudioDecoder {
public:explicit DispatchAudioDecoder(std::shared_ptr<Resources> r):AudioDecoder(std::move(r),std::make_shared<RecordingDense>()){}
};
}
#define AudioDecoder DispatchAudioDecoder
#include "../src/tts_audio_decode.cpp"
#undef AudioDecoder
