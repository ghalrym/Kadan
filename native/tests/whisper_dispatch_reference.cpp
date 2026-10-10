// Reuse diagnostic transport with an injected CPU backend for randomized parity.
#include "kadan/whisper.hpp"
#include "dense_reference.hpp"
namespace kadan::stt {
class DispatchWhisper:public Whisper {
public:explicit DispatchWhisper(std::shared_ptr<Resources> r):Whisper(std::move(r),std::make_shared<RecordingDense>()){}
};
}
#define Whisper DispatchWhisper
#include "../src/whisper_network.cpp"
#undef Whisper
