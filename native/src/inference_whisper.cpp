#include "kadan/inference_engines.hpp"
#include "kadan/stt.hpp"
#include "kadan/whisper.hpp"
#include "kadan/whisper_tokens.hpp"
#include <fstream>

namespace kadan::serving {
namespace {
void read_floats(const std::string &path, std::span<float> output) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(bool(file) && file.tellg() == std::streamoff(output.size_bytes()),
          "whisper_input_file");
  file.seekg(0);
  file.read(reinterpret_cast<char *>(output.data()), output.size_bytes());
  require(bool(file), "whisper_input_read");
}

class WhisperEngine final : public InferenceEngine {
public:
  WhisperEngine(std::shared_ptr<Resources> resources, Progress progress)
      : resources_(std::move(resources)), progress_(std::move(progress)) {}

  void load(const InferenceRequest &request, const Footprint &available,
            const std::atomic_bool &cancel) override {
    if (available[0] < 1024ULL * 1024 * 1024)
      throw AdmissionWait("whisper_host_headroom");
    const auto input = Json::parse(request.payload);
    const auto root = input.at("checkpoint").get<std::string>();
    const auto assets = input.at("assets").get<std::string>();
    std::ifstream file(root + "/dimensions.txt");
    auto &dimensions = dimensions_;
    file >> dimensions.mels >> dimensions.audio_context >>
        dimensions.audio_state >> dimensions.audio_heads >>
        dimensions.audio_layers >> dimensions.vocabulary >>
        dimensions.text_context >> dimensions.text_state >>
        dimensions.text_heads >> dimensions.text_layers;
    std::string extra;
    require(bool(file) && !(file >> extra) &&
                dimensions.audio_context == 1500 &&
                (dimensions.mels == 80 || dimensions.mels == 128),
            "whisper_dimensions_file");
    compute_ = std::make_unique<WorkerCompute>(media_plan(available),
                                               Workload::speech, resources_);
    workspace_ = std::make_unique<HostAllocation>(resources_, Workload::speech,
                                                  256 * 1024 * 1024);
    model_ = std::make_unique<stt::Whisper>(resources_, compute_->compute);
    model_->load(root.c_str(), "model.safetensors", dimensions, cancel);
    tokens_ = std::make_unique<stt::WhisperTokens>(resources_);
    tokens_->load((assets + "/english.tokens").c_str(), dimensions.vocabulary,
                  cancel);
    require(tokens_->prompt().size() <= dimensions.text_context,
            "whisper_prompt_context");
    frontend_ = std::make_unique<stt::LogMel>(resources_);
    HostAllocation filter_memory(resources_, Workload::speech,
                                 dimensions.mels * 201 * 4);
    std::vector<float> filters(dimensions.mels * 201);
    read_floats(assets + "/mel-" + std::to_string(dimensions.mels) + ".f32",
                filters);
    frontend_->load(dimensions.mels, filters, cancel);
  }

  void prepare(const InferenceRequest &request, const Footprint &available,
               const std::atomic_bool &cancel) override {
    check_cancel(cancel);
    if (request.operation == Operation::load)
      return;
    const auto input = Json::parse(request.payload);
    input_ = input.at("input");
    std::ifstream file(input_, std::ios::binary | std::ios::ate);
    require(bool(file), "whisper_pcm_file");
    auto bytes = file.tellg();
    require(bytes >= 0 && bytes % 4 == 0 && bytes <= 480000 * 4,
            "whisper_pcm_bound");
    samples_ = std::size_t(bytes) / 4;
    max_tokens_ = std::min<std::size_t>(224, dimensions_.text_context -
                                                 tokens_->prompt().size() + 1);
    const Bytes scratch =
        4 * (samples_ + dimensions_.mels * 3000 +
             dimensions_.audio_context * dimensions_.audio_state +
             max_tokens_) +
        stt::WhisperTokens::max_text_bytes;
    if (available[0] < scratch)
      throw AdmissionWait("whisper_request_headroom");
    request_memory_ =
        std::make_unique<HostAllocation>(resources_, Workload::speech, scratch);
  }

  std::string execute(const InferenceRequest &request,
                      const std::atomic_bool &cancel) override {
    if (request.operation == Operation::load)
      return "{}";
    std::string text;
    {
      std::vector<float> pcm(samples_), mel(dimensions_.mels * 3000),
          encoded(dimensions_.audio_context * dimensions_.audio_state);
      std::vector<std::uint32_t> generated(max_tokens_);
      read_floats(input_, pcm);
      frontend_->execute_window(pcm, mel, cancel);
      model_->encode(mel, encoded, cancel, progress_);
      const auto count =
          model_->greedy(tokens_->prompt(), encoded, generated, tokens_->eos(),
                         tokens_->suppressed(), cancel, progress_,
                         tokens_->first_suppressed());
      require(count && generated[count - 1] == tokens_->eos(),
              "whisper_token_limit");
      text = tokens_->decode(std::span(generated).first(count));
    }
    compute_->idle();
    request_memory_.reset();
    return Json{{"text", text}}.dump();
  }

  void close() override {
    request_memory_.reset();
    frontend_.reset();
    tokens_.reset();
    model_.reset();
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
  std::unique_ptr<stt::Whisper> model_;
  std::unique_ptr<stt::WhisperTokens> tokens_;
  std::unique_ptr<stt::LogMel> frontend_;
  stt::WhisperDimensions dimensions_{};
  std::string input_;
  std::size_t samples_ = 0, max_tokens_ = 0;
};
} // namespace

std::unique_ptr<InferenceEngine>
whisper_engine(std::shared_ptr<Resources> resources, Progress progress) {
  return std::make_unique<WhisperEngine>(std::move(resources),
                                         std::move(progress));
}
} // namespace kadan::serving
