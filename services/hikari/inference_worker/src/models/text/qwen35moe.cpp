// Qwen3.6 35B-A3B (GGUF qwen35moe): llama.cpp src/models/qwen35moe.cpp at
// 6420cf5dd588c68e48657a0bbd82bb3bd5ac7209. Text only; MTP is not executed.
#include <cmath>
#include <string>
#include <variant>

#include "gguf_model.hpp"
#include "qwen_layers.hpp"

namespace kadan::text {
namespace {
constexpr std::size_t hidden = 2048, layers = 40, experts = 256, top_k = 8,
                      expert_ff = 512, vocabulary_size = 248320,
                      context_length = 262144;
constexpr float eps = 1e-6f;
const DeltaNetSpec delta_spec{hidden, 16, 32, 128, 4, eps, false};
const AttentionSpec attention_spec{hidden, 16, 2, 256, 64, 1e7f, eps};
const MoeSpec moe_spec{hidden, experts, top_k, expert_ff, expert_ff};

bool full_attention(std::size_t layer) {
    return (layer + 1) % 4 == 0;
}

std::string prefix(std::size_t layer) {
    return "blk." + std::to_string(layer) + ".";
}

template <class Metadata> void validate(const Metadata& m) {
    need(m.string("general.architecture") == "qwen35moe", "qwen35moe_architecture");
    for (const auto& [key, expected] :
         std::initializer_list<std::pair<const char*, std::uint64_t>>{
             {"block_count", layers},
             {"context_length", context_length},
             {"embedding_length", hidden},
             {"attention.head_count", attention_spec.heads},
             {"attention.head_count_kv", attention_spec.kv_heads},
             {"attention.key_length", attention_spec.dim},
             {"attention.value_length", attention_spec.dim},
             {"rope.dimension_count", attention_spec.rotary},
             {"expert_count", experts},
             {"expert_used_count", top_k},
             {"expert_feed_forward_length", expert_ff},
             {"expert_shared_feed_forward_length", expert_ff},
             {"ssm.conv_kernel", delta_spec.conv},
             {"ssm.state_size", delta_spec.dim},
             {"ssm.group_count", delta_spec.key_heads},
             {"ssm.time_step_rank", delta_spec.value_heads},
             {"ssm.inner_size", delta_spec.value_heads * delta_spec.dim},
             {"full_attention_interval", 4}
         })
        need(
            m.integer(std::string("qwen35moe.") + key) == expected,
            "qwen35moe_dimensions"
        );
    need(
        std::abs(m.number("qwen35moe.rope.freq_base") - 1e7) < 1
            && std::abs(m.number("qwen35moe.attention.layer_norm_rms_epsilon") - eps)
                   < 1e-12,
        "qwen35moe_arithmetic"
    );
    const auto sections = m.integers("qwen35moe.rope.dimension_sections");
    need(
        sections.size() == 4 && sections[0] == 11 && sections[1] == 11
            && sections[2] == 10 && sections[3] == 0,
        "qwen35moe_rope_sections"
    );
    // Absent keys select llama.cpp's defaults: no routed weight scale, no MTP,
    // and the interval-derived recurrent schedule.
    need(
        !m.has("qwen35moe.expert_weights_scale")
            && !m.has("qwen35moe.nextn_predict_layers")
            && !m.has("qwen35moe.attention.recurrent_layers")
            && m.string("tokenizer.ggml.model") == "gpt2"
            && m.string("tokenizer.ggml.pre") == "qwen35"
            && m.integer("tokenizer.ggml.eos_token_id") == 248046
            && m.integer("tokenizer.ggml.padding_token_id") == 248044,
        "qwen35moe_tokenizer"
    );
}

struct Layer {
    Vec norm, post_norm;
    std::variant<DeltaNetLayer, AttentionLayer> attention;
    MoeLayer moe;
};

class Qwen35Moe final : public Model {
public:
    Qwen35Moe(const GgufWeights& weights, DenseCompute& compute, std::size_t context)
        : w_(weights), compute_(compute), context_(context),
          output_norm_(weights.read("output_norm.weight")) {
        need(context && context <= context_length, "qwen35moe_context");
        for (std::size_t i = 0; i < layers; ++i) {
            const auto p = prefix(i);
            layers_.push_back(
                {w_.read(p + "attn_norm.weight"),
                 w_.read(p + "post_attention_norm.weight"),
                 full_attention(i)
                     ? std::variant<DeltaNetLayer, AttentionLayer>(
                           std::in_place_type<AttentionLayer>,
                           w_,
                           p,
                           attention_spec,
                           context
                       )
                     : std::variant<DeltaNetLayer, AttentionLayer>(
                           std::in_place_type<DeltaNetLayer>, w_, p, delta_spec
                       ),
                 MoeLayer(w_, p, moe_spec)}
            );
        }
    }

    std::size_t vocabulary() const override {
        return vocabulary_size;
    }

    std::size_t device_state_bytes() const override {
        return 10
                   * AttentionLayer::state_bytes(
                       attention_spec, std::max<std::size_t>(context_, 256)
                   )
               + 30 * DeltaNetLayer::device_bytes(delta_spec);
    }

    bool eos(unsigned token) const override {
        return token == 248046 || token == 248044;
    }

    void reset() override {
        for (auto& layer : layers_)
            if (auto* delta = std::get_if<DeltaNetLayer>(&layer.attention))
                delta->reset();
    }

    Vec forward(
        std::span<const unsigned> tokens,
        std::size_t position,
        bool logits,
        const Progress& progress
    ) override {
        need(
            !tokens.empty() && position + tokens.size() <= context_,
            "qwen35moe_context_exceeded"
        );
        Vec x(tokens.size() * hidden);
        for (std::size_t row = 0; row < tokens.size(); ++row) {
            need(tokens[row] < vocabulary_size, "qwen35moe_token");
            w_.rows(
                "token_embd.weight",
                tokens[row],
                std::span(x).subspan(row * hidden, hidden)
            );
        }
        for (std::size_t i = 0; i < layers; ++i) {
            auto& layer = layers_[i];
            Vec h = x;
            rms_norm(h, layer.norm, eps);
            auto* attention = std::get_if<AttentionLayer>(&layer.attention);
            const auto mixed =
                attention
                    ? attention->forward(compute_, w_, h, position)
                    : std::get<DeltaNetLayer>(layer.attention).forward(compute_, w_, h);
            for (std::size_t j = 0; j < x.size(); ++j)
                x[j] += mixed[j];
            h = x;
            rms_norm(h, layer.post_norm, eps);
            const auto moe = layer.moe.forward(compute_, w_, h);
            for (std::size_t j = 0; j < x.size(); ++j)
                x[j] += moe[j];
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
    std::vector<Layer> layers_;
};
}  // namespace

void validate_qwen35moe(const checkpoint::GgufMetadata& m) {
    validate(m);
}

void validate_qwen35moe(const checkpoint::GgufCheckpoint& m) {
    validate(m);
}

void bind_qwen35moe(GgufWeights& w) {
    validate_qwen35moe(w.gguf());
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
        bind_moe(w, p, moe_spec);
    }
    w.finish();
}

std::size_t qwen35moe_state_bytes(std::size_t context) {
    return 30 * DeltaNetLayer::state_bytes(delta_spec)
           + 10 * AttentionLayer::state_bytes(attention_spec, context);
}

std::unique_ptr<Model>
qwen35moe_model(const GgufWeights& w, DenseCompute& compute, std::size_t context) {
    return std::make_unique<Qwen35Moe>(w, compute, context);
}
}  // namespace kadan::text
