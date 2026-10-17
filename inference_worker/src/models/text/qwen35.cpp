// Qwen3.8 27B (GGUF qwen35): llama.cpp src/models/qwen35.cpp at
// 6420cf5dd588c68e48657a0bbd82bb3bd5ac7209. Text only; MTP is not executed.
#include <cmath>
#include <string>

#include "gguf_model.hpp"
#include "qwen_layers.hpp"

namespace kadan::text {
namespace {
constexpr std::size_t hidden = 5120, layers = 64, ff = 17408, vocabulary_size = 248320,
                      context_length = 262144;
constexpr float eps = 1e-6f;
const DeltaNetSpec delta_spec{hidden, 16, 48, 128, 4, eps, false};
const AttentionSpec attention_spec{hidden, 24, 4, 256, 64, 1e7f, eps};

bool full_attention(std::size_t layer) {
    return (layer + 1) % 4 == 0;
}

std::string prefix(std::size_t layer) {
    return "blk." + std::to_string(layer) + ".";
}

template <class Metadata> void validate(const Metadata& m) {
    need(m.string("general.architecture") == "qwen35", "qwen35_architecture");
    for (const auto& [key, expected] :
         std::initializer_list<std::pair<const char*, std::uint64_t>>{
             {"block_count", layers},
             {"context_length", context_length},
             {"embedding_length", hidden},
             {"feed_forward_length", ff},
             {"attention.head_count", attention_spec.heads},
             {"attention.head_count_kv", attention_spec.kv_heads},
             {"attention.key_length", attention_spec.dim},
             {"attention.value_length", attention_spec.dim},
             {"rope.dimension_count", attention_spec.rotary},
             {"ssm.conv_kernel", delta_spec.conv},
             {"ssm.state_size", delta_spec.dim},
             {"ssm.group_count", delta_spec.key_heads},
             {"ssm.time_step_rank", delta_spec.value_heads},
             {"ssm.inner_size", delta_spec.value_heads * delta_spec.dim},
             {"full_attention_interval", 4}
         })
        need(m.integer(std::string("qwen35.") + key) == expected, "qwen35_dimensions");
    need(
        std::abs(m.number("qwen35.rope.freq_base") - 1e7) < 1
            && std::abs(m.number("qwen35.attention.layer_norm_rms_epsilon") - eps)
                   < 1e-12,
        "qwen35_arithmetic"
    );
    const auto sections = m.integers("qwen35.rope.dimension_sections");
    need(
        sections.size() == 4 && sections[0] == 11 && sections[1] == 11
            && sections[2] == 10 && sections[3] == 0,
        "qwen35_rope_sections"
    );
    const auto recurrent = m.booleans("qwen35.attention.recurrent_layers");
    need(recurrent.size() == layers, "qwen35_layer_schedule");
    for (std::size_t i = 0; i < layers; ++i)
        need(recurrent[i] == !full_attention(i), "qwen35_layer_schedule");
    need(
        !m.has("qwen35.nextn_predict_layers")
            && m.string("tokenizer.ggml.model") == "gpt2"
            && m.string("tokenizer.ggml.pre") == "qwen35"
            && m.integer("tokenizer.ggml.eos_token_id") == 248046
            && m.integer("tokenizer.ggml.padding_token_id") == 248044,
        "qwen35_tokenizer"
    );
}

class Qwen35 final : public Model {
public:
    Qwen35(const GgufWeights& weights, DenseCompute& compute, std::size_t context)
        : w_(weights), compute_(compute), context_(context),
          output_norm_(weights.read("output_norm.weight")) {
        need(context && context <= context_length, "qwen35_context");
        for (std::size_t i = 0; i < layers; ++i) {
            const auto p = prefix(i);
            norms_.push_back(w_.read(p + "attn_norm.weight"));
            post_norms_.push_back(w_.read(p + "post_attention_norm.weight"));
            if (full_attention(i))
                attention_.emplace_back(w_, p, attention_spec, context);
            else
                delta_.emplace_back(w_, p, delta_spec);
        }
    }

    std::size_t vocabulary() const override {
        return vocabulary_size;
    }

    bool eos(unsigned token) const override {
        return token == 248046 || token == 248044;
    }

    void reset() override {
        for (auto& layer : delta_)
            layer.reset();
    }

    Vec forward(
        std::span<const unsigned> tokens,
        std::size_t position,
        bool logits,
        const Progress& progress
    ) override {
        need(
            !tokens.empty() && position + tokens.size() <= context_,
            "qwen35_context_exceeded"
        );
        Vec x(tokens.size() * hidden);
        for (std::size_t row = 0; row < tokens.size(); ++row) {
            need(tokens[row] < vocabulary_size, "qwen35_token");
            w_.rows(
                "token_embd.weight",
                tokens[row],
                std::span(x).subspan(row * hidden, hidden)
            );
        }
        std::size_t next_delta = 0, next_attention = 0;
        for (std::size_t i = 0; i < layers; ++i) {
            const auto p = prefix(i);
            Vec h = x;
            rms_norm(h, norms_[i], eps);
            const auto mixed =
                full_attention(i)
                    ? attention_[next_attention++].forward(compute_, w_, h, position)
                    : delta_[next_delta++].forward(compute_, w_, h);
            for (std::size_t j = 0; j < x.size(); ++j)
                x[j] += mixed[j];
            h = x;
            rms_norm(h, post_norms_[i], eps);
            auto gate = w_.project(compute_, p + "ffn_gate.weight", h);
            const auto up = w_.project(compute_, p + "ffn_up.weight", h);
            for (std::size_t j = 0; j < gate.size(); ++j)
                gate[j] = silu(gate[j]) * up[j];
            const auto down = w_.project(compute_, p + "ffn_down.weight", gate);
            for (std::size_t j = 0; j < x.size(); ++j)
                x[j] += down[j];
            if (progress)
                progress("layer", i + 1);
        }
        if (!logits)
            return {};
        Vec last(x.end() - hidden, x.end());
        rms_norm(last, output_norm_, eps);
        return w_.project(compute_, "output.weight", last);
    }

private:
    const GgufWeights& w_;
    DenseCompute& compute_;
    std::size_t context_;
    Vec output_norm_;
    std::vector<Vec> norms_, post_norms_;
    std::vector<DeltaNetLayer> delta_;
    std::vector<AttentionLayer> attention_;
};
}  // namespace

void validate_qwen35(const checkpoint::GgufMetadata& m) {
    validate(m);
}

void validate_qwen35(const checkpoint::GgufCheckpoint& m) {
    validate(m);
}

void bind_qwen35(GgufWeights& w) {
    validate_qwen35(w.gguf());
    w.matrix("token_embd.weight", vocabulary_size, hidden);
    w.matrix("output.weight", vocabulary_size, hidden);
    w.vector("output_norm.weight", {hidden});
    for (std::size_t i = 0; i < layers; ++i) {
        const auto p = prefix(i);
        w.vector(p + "attn_norm.weight", {hidden});
        w.vector(p + "post_attention_norm.weight", {hidden});
        if (full_attention(i))
            bind_attention(w, p, attention_spec);
        else
            bind_delta_net(w, p, delta_spec);
        w.matrix(p + "ffn_gate.weight", ff, hidden);
        w.matrix(p + "ffn_up.weight", ff, hidden);
        w.matrix(p + "ffn_down.weight", hidden, ff);
    }
    w.finish();
}

std::size_t qwen35_state_bytes(std::size_t context) {
    return 48 * DeltaNetLayer::state_bytes(delta_spec)
           + 16 * AttentionLayer::state_bytes(attention_spec, context);
}

std::unique_ptr<Model>
qwen35_model(const GgufWeights& w, DenseCompute& compute, std::size_t context) {
    return std::make_unique<Qwen35>(w, compute, context);
}
}  // namespace kadan::text
