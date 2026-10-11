#include "kadan/inference_engines.hpp"
#include "kadan/tts_audio.hpp"
#include "kadan/tts_controls.hpp"
#include "kadan/tts_generator.hpp"
#include "kadan/tts_tokenizer.hpp"
#include <array>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>

namespace kadan::serving {
namespace {
void publish_audio(const std::string &path, std::span<const float> samples,
                   const std::atomic_bool &cancel) {
  std::vector<std::uint8_t> bytes(44 + samples.size() * 2);
  auto word = [&](std::size_t offset, std::uint32_t value, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i)
      bytes[offset + i] = std::uint8_t(value >> (8 * i));
  };
  auto tag = [&](std::size_t offset, const char *text) {
    std::copy_n(text, 4, bytes.begin() + offset);
  };
  tag(0, "RIFF");
  word(4, 36 + samples.size() * 2, 4);
  tag(8, "WAVE");
  tag(12, "fmt ");
  word(16, 16, 4);
  word(20, 1, 2);
  word(22, 1, 2);
  word(24, 24000, 4);
  word(28, 48000, 4);
  word(32, 2, 2);
  word(34, 16, 2);
  tag(36, "data");
  word(40, samples.size() * 2, 4);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    require(std::isfinite(samples[i]), "tts_nonfinite_audio");
    auto value = std::clamp(std::lround(samples[i] * 32768), -32768L, 32767L);
    word(44 + 2 * i, std::uint16_t(value), 2);
  }
  check_cancel(cancel);
  int file = open(path.c_str(),
                  O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0600);
  require(file >= 0, "tts_output_create");
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    auto count = write(file, bytes.data() + offset, bytes.size() - offset);
    if (count <= 0 || cancel.load()) {
      close(file);
      unlink(path.c_str());
      throw std::runtime_error("tts_output_write");
    }
    offset += count;
  }
  if (close(file) != 0) {
    unlink(path.c_str());
    throw std::runtime_error("tts_output_close");
  }
}

class SpeechEngine final : public InferenceEngine {
public:
  SpeechEngine(std::shared_ptr<Resources> resources, Progress progress)
      : resources_(std::move(resources)), progress_(std::move(progress)) {}

  void load(const InferenceRequest &request, const Footprint &available,
            const std::atomic_bool &cancel) override {
    if (available[0] < 2ULL * 1024 * 1024 * 1024)
      throw AdmissionWait("speech_host_headroom");
    auto input = Json::parse(request.payload);
    const auto root = input.at("checkpoint").get<std::string>();
    compute_ = std::make_unique<WorkerCompute>(media_plan(available),
                                               Workload::tts, resources_);
    workspace_ = std::make_unique<HostAllocation>(resources_, Workload::tts,
                                                  256 * 1024 * 1024);
    controls_ = std::make_unique<tts::VoiceControls>(root + "/config.json");
    tokenizer_ = std::make_unique<tts::TextTokenizer>(resources_);
    tokenizer_->load(input.at("tokenizer").get<std::string>(), cancel);
    generator_ =
        std::make_unique<tts::CodeGenerator>(resources_, compute_->compute);
    generator_->load(root.c_str(), "model.safetensors", {}, cancel);
    decoder_ =
        std::make_unique<tts::AudioDecoder>(resources_, compute_->compute);
    decoder_->load((root + "/speech_tokenizer").c_str(), "model.safetensors",
                   {}, cancel);
  }

  void prepare(const InferenceRequest &request, const Footprint &available,
               const std::atomic_bool &cancel) override {
    check_cancel(cancel);
    if (request.operation == Operation::load)
      return;
    constexpr Bytes scratch =
        2 * 1024 * 1024 + 2 * 2048 * 4 + 300 * 16 * 4 + 300 * 1920 * 6 + 44;
    if (available[0] < scratch)
      throw AdmissionWait("speech_request_headroom");
    request_memory_ =
        std::make_unique<HostAllocation>(resources_, Workload::tts, scratch);
    const auto input = Json::parse(request.payload);
    const auto &body = input.at("request");
    script_ = body.at("script");
    instruction_ = body.value("instruction", "");
    require(!script_.empty() && script_.size() <= 32000 &&
                instruction_.size() <= 8000,
            "speech_request_bounds");
    voice_ = controls_->resolve(body.at("speaker"), body.at("language"));
    output_ = input.at("output");
  }

  std::string execute(const InferenceRequest &request,
                      const std::atomic_bool &cancel) override {
    if (request.operation == Operation::load)
      return "{}";
    std::size_t frames = 0;
    {
      std::array<std::uint32_t, 2048> tokens, instruction;
      const auto count = tokenizer_->encode(script_, tokens, cancel);
      std::size_t instruction_count = 0;
      if (!instruction_.empty())
        instruction_count = tokenizer_->encode(
            "<|im_start|>user\n" + instruction_ + "<|im_end|>\n", instruction,
            cancel);
      std::vector<std::uint32_t> codes(300 * 16);
      const auto generated = generator_->generate(
          std::span(tokens).first(count), voice_, codes, cancel, progress_, 2,
          1.05f, std::span(instruction).first(instruction_count));
      require(generated.stopped && generated.frames > 0,
              "tts_incomplete_generation");
      std::vector<float> audio(generated.frames * 1920);
      decoder_->decode(std::span(codes).first(generated.frames * 16), audio,
                       cancel, progress_);
      publish_audio(output_, audio, cancel);
      frames = generated.frames;
    }
    compute_->idle();
    request_memory_.reset();
    return Json{{"output", output_}, {"frames", frames}}.dump();
  }

  void close() override {
    request_memory_.reset();
    decoder_.reset();
    generator_.reset();
    tokenizer_.reset();
    controls_.reset();
    workspace_.reset();
    if (compute_)
      compute_->park();
    compute_.reset();
  }

private:
  std::shared_ptr<Resources> resources_;
  Progress progress_;
  std::unique_ptr<WorkerCompute> compute_;
  std::unique_ptr<HostAllocation> workspace_, request_memory_;
  std::unique_ptr<tts::VoiceControls> controls_;
  std::unique_ptr<tts::TextTokenizer> tokenizer_;
  std::unique_ptr<tts::CodeGenerator> generator_;
  std::unique_ptr<tts::AudioDecoder> decoder_;
  tts::VoicePrompt voice_;
  std::string script_, instruction_, output_;
};
} // namespace

std::unique_ptr<InferenceEngine>
speech_engine(std::shared_ptr<Resources> resources, Progress progress) {
  return std::make_unique<SpeechEngine>(std::move(resources),
                                        std::move(progress));
}
} // namespace kadan::serving
