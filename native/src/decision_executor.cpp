#include "kadan/decision_executor.hpp"
#include "decision_tokenizer.hpp"
#include "kadan/checkpoint.hpp"
#include "kadan/weight_backing.hpp"
#include <algorithm>
#include <bit>
#include <cblas.h>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <set>
#include <unordered_map>

namespace kadan::decision {
namespace {
constexpr Bytes MiB = 1024 * 1024, metadata_bytes = 128 * MiB, scratch_bytes = 256 * MiB;
Footprint host(const std::shared_ptr<Resources> &r, Bytes bytes) {
    auto f = r->snapshot().capacity;
    std::fill(f.begin(), f.end(), 0);
    f[0] = bytes;
    return f;
}
struct Reservation {
    std::shared_ptr<Resources> r;
    Handle h;
    Reservation(std::shared_ptr<Resources> r_, Bytes n)
        : r(r_), h(r->reserve(Workload::decision, host(r, n))) {}
    ~Reservation() {
        if (h)
            r->released(h);
    }
};
struct Pin {
    std::shared_ptr<Resources> r;
    Handle h;
    Pin(std::shared_ptr<Resources> r_, Handle h_) : r(r_), h(h_) { r->pin(h); }
    ~Pin() { r->unpin(h); }
};
using Matrix = std::vector<float>;
struct Tensor {
    std::vector<std::uint64_t> shape;
    Matrix values;
};
float half(std::uint16_t bits) {
    int e = (bits >> 10) & 31, f = bits & 1023;
    check(e != 31, "decision_nonfinite_weight");
    float v = e ? std::ldexp(float(1024 + f), e - 25) : std::ldexp(float(f), -24);
    return bits & 32768 ? -v : v;
}
float gelu(float x) { return .5f * x * (1.f + std::erf(x / std::sqrt(2.f))); }
void add(Matrix &a, const Matrix &b) {
    check(a.size() == b.size(), "decision_add_shape");
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] += b[i];
}
void norm(Matrix &x, int width, const Tensor &weight, const Tensor *bias, float epsilon) {
    for (std::size_t row = 0; row < x.size(); row += width) {
        double sum = 0;
        for (int c = 0; c < width; ++c)
            sum += x[row + c];
        double mean = sum / width, var = 0;
        for (int c = 0; c < width; ++c) {
            double d = x[row + c] - mean;
            var += d * d;
        }
        float inv = 1.f / std::sqrt(float(var / width) + epsilon);
        for (int c = 0; c < width; ++c)
            x[row + c] = (x[row + c] - float(mean)) * inv * weight.values[c] +
                         (bias ? bias->values[c] : 0.f);
    }
}
Matrix linear(const Matrix &x, int input, const Tensor &w, const Tensor *b = nullptr) {
    const int rows = int(x.size() / input), output = int(w.shape[0]);
    Matrix result(std::size_t(rows) * output);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, output, input, 1.f, x.data(), input,
                w.values.data(), input, 0.f, result.data(), output);
    if (b)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < output; ++c)
                result[std::size_t(r) * output + c] += b->values[c];
    return result;
}
std::vector<float> softmax(std::vector<float> x) {
    float peak = *std::max_element(x.begin(), x.end()), sum = 0;
    for (float &v : x) {
        v = std::exp(v - peak);
        sum += v;
    }
    check(std::isfinite(sum) && sum > 0, "decision_nonfinite_output");
    for (float &v : x)
        v /= sum;
    return x;
}
Matrix attention(Matrix qkv, int rows, int hidden, int heads, int window, float theta,
                 const std::atomic_bool &cancel) {
    int dim = hidden / heads;
    if (theta > 0)
        for (int row = 0; row < rows; ++row)
            for (int part = 0; part < 2; ++part)
                for (int head = 0; head < heads; ++head)
                    for (int d = 0; d < dim / 2; ++d) {
                        float angle = float(row) / std::pow(theta, float(2 * d) / dim),
                              cs = std::cos(angle), sn = std::sin(angle);
                        auto i = std::size_t(row) * 3 * hidden + part * hidden + head * dim + d;
                        float a = qkv[i], b = qkv[i + dim / 2];
                        qkv[i] = a * cs - b * sn;
                        qkv[i + dim / 2] = b * cs + a * sn;
                    }
    Matrix output(std::size_t(rows) * hidden), scores(std::size_t(rows) * rows),
        values(std::size_t(rows) * dim);
    for (int head = 0; head < heads; ++head) {
        cancel_check(cancel);
        const float *q = qkv.data() + head * dim;
        const float *k = q + hidden;
        const float *v = k + hidden;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, rows, dim,
                    1.f / std::sqrt(float(dim)), q, 3 * hidden, k, 3 * hidden, 0.f, scores.data(),
                    rows);
        for (int row = 0; row < rows; ++row) {
            float peak = -INFINITY, sum = 0;
            for (int col = 0; col < rows; ++col) {
                auto &s = scores[std::size_t(row) * rows + col];
                if (window >= 0 && std::abs(col - row) > window)
                    s = -INFINITY;
                peak = std::max(peak, s);
            }
            for (int col = 0; col < rows; ++col) {
                auto &s = scores[std::size_t(row) * rows + col];
                s = std::exp(s - peak);
                sum += s;
            }
            for (int col = 0; col < rows; ++col)
                scores[std::size_t(row) * rows + col] /= sum;
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, dim, rows, 1.f, scores.data(),
                    rows, v, 3 * hidden, 0.f, values.data(), dim);
        for (int row = 0; row < rows; ++row)
            std::copy_n(values.data() + std::size_t(row) * dim, dim,
                        output.data() + std::size_t(row) * hidden + head * dim);
    }
    return output;
}
std::string string_field(const Json &j, const char *key, bool empty = false) {
    check(j.contains(key) && j.at(key).is_string(), "decision_string_field");
    auto s = j.at(key).get<std::string>();
    check(s.size() <= 16384 && (empty || s.find_first_not_of(" \t\r\n") != std::string::npos),
          "decision_empty_or_long_text");
    return s;
}
int integer(const Json &object, const char *field) {
    const auto &value = object.at(field);
    check(value.is_number_integer() && value >= 0 && value <= 65536, "decision_integer_config");
    return value.get<int>();
}
float temperature(const Json &value) {
    if (!value.is_number())
        return 1.f;
    return std::clamp(value.get<float>(), .5f, 5.f);
}
double rounded(double v) { return std::nearbyint(v * 10000.) / 10000.; }
} // namespace
struct Executor::Impl {
    Json config, encoder;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unordered_map<std::string, Tensor> weights;
    int hidden = 0, intermediate = 0, layers = 0, heads = 0, head_layers = 0, vocab = 0,
        max_len = 0, head_len = 0, window = 0;
    float eps = 1e-5f, global_theta = 160000.f, local_theta = 10000.f;
    const Tensor &w(const std::string &key) const { return weights.at(key); }
    Matrix normed(Matrix x, const std::string &prefix, bool bias = false,
                  float epsilon = 1e-5f) const {
        norm(x, hidden, w(prefix + ".weight"), bias ? &w(prefix + ".bias") : nullptr, epsilon);
        return x;
    }
    Matrix forward(const std::vector<int> &ids, int type, const std::atomic_bool &cancel,
                   const Observer &observer) const {
        int rows = int(ids.size());
        Matrix x(std::size_t(rows) * hidden);
        const auto &embeddings = w("encoder.embeddings.tok_embeddings.weight").values;
        for (int row = 0; row < rows; ++row) {
            check(ids[row] >= 0 && ids[row] < vocab, "decision_token_id");
            std::copy_n(embeddings.data() + std::size_t(ids[row]) * hidden, hidden,
                        x.data() + std::size_t(row) * hidden);
        }
        norm(x, hidden, w("encoder.embeddings.norm.weight"), nullptr, eps);
        for (int layer = 0; layer < layers; ++layer) {
            cancel_check(cancel);
            if (observer)
                observer("encoder");
            cancel_check(cancel);
            const auto p = "encoder.layers." + std::to_string(layer);
            bool local = encoder.at("layer_types").at(layer) == "sliding_attention";
            Matrix n = layer ? normed(x, p + ".attn_norm", false, eps) : x;
            auto qkv = linear(n, hidden, w(p + ".attn.Wqkv.weight"));
            auto a = attention(std::move(qkv), rows, hidden, heads, local ? window : -1,
                               local ? local_theta : global_theta, cancel);
            add(x, linear(a, hidden, w(p + ".attn.Wo.weight")));
            auto gate =
                linear(normed(x, p + ".mlp_norm", false, eps), hidden, w(p + ".mlp.Wi.weight"));
            Matrix activation(std::size_t(rows) * intermediate);
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < intermediate; ++c)
                    activation[std::size_t(r) * intermediate + c] =
                        gelu(gate[std::size_t(r) * 2 * intermediate + c]) *
                        gate[std::size_t(r) * 2 * intermediate + intermediate + c];
            cancel_check(cancel);
            add(x, linear(activation, intermediate, w(p + ".mlp.Wo.weight")));
        }
        norm(x, hidden, w("encoder.final_norm.weight"), nullptr, eps);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < hidden; ++c)
                x[std::size_t(r) * hidden + c] += w("type_emb.weight").values[type * hidden + c];
        for (int layer = 0; layer < head_layers; ++layer) {
            cancel_check(cancel);
            if (observer)
                observer("head");
            cancel_check(cancel);
            auto p = "head.layers." + std::to_string(layer);
            auto qkv =
                linear(normed(x, p + ".norm1", true), hidden, w(p + ".self_attn.in_proj_weight"),
                       &w(p + ".self_attn.in_proj_bias"));
            auto a =
                attention(std::move(qkv), rows, hidden, std::max(1, hidden / 64), -1, 0, cancel);
            add(x, linear(a, hidden, w(p + ".self_attn.out_proj.weight"),
                          &w(p + ".self_attn.out_proj.bias")));
            auto ff = linear(normed(x, p + ".norm2", true), hidden, w(p + ".linear1.weight"),
                             &w(p + ".linear1.bias"));
            for (float &v : ff)
                v = std::max(v, 0.f); // TransformerEncoderLayer defaults to ReLU, not GELU.
            add(x, linear(ff, 4 * hidden, w(p + ".linear2.weight"), &w(p + ".linear2.bias")));
        }
        return x;
    }
};
Executor::Executor(std::shared_ptr<Resources> r) : resources_(std::move(r)) {
    check(bool(resources_), "decision_resources");
}
Executor::~Executor() { unload(); }
bool Executor::loaded() const { return bool(impl_); }
void Executor::unload() {
    if (!resident_)
        return;
    check(resources_->reservation(resident_).pins == 0, "decision_busy");
    if (!borrowed_)
        resources_->begin_eviction(resident_);
    impl_.reset();
    if (!borrowed_)
        resources_->released(resident_);
    resident_ = 0;
    borrowed_ = false;
}
void Executor::load(const std::string &root, const std::atomic_bool &cancel, Handle envelope,
                    const Observer &observer) {
    check(!loaded(), "decision_already_loaded");
    cancel_check(cancel);
    if (envelope) {
        auto r = resources_->reservation(envelope);
        check(r.workload == Workload::decision && r.state == State::loading && r.pins == 0 &&
                  r.bytes[0] >= envelope_bytes,
              "decision_admission");
    }
    std::unique_ptr<Reservation> admission;
    if (!envelope)
        admission = std::make_unique<Reservation>(resources_, metadata_bytes);
    auto next = std::make_unique<Impl>();
    auto &m = *next;
    m.config = read_json(root + "/rl_agent_config.json", 64 * 1024);
    m.encoder = read_json(root + "/encoder/config.json", 64 * 1024);
    const auto &c = m.encoder;
    check(m.config.at("act_costs").is_object() && m.config.at("act_costs").size() == 1,
          "decision_action_config");
    check(m.config.at("temperature").is_array() && m.config.at("temperature").size() == 3,
          "decision_temperature_config");
    check(!m.config.contains("temperature_by_options") ||
              m.config.at("temperature_by_options").is_object(),
          "decision_temperature_config");
    check(integer(c, "local_attention") % 2 == 0, "decision_window");
    check(c.at("model_type") == "modernbert" && !c.at("attention_bias").get<bool>() &&
              !c.at("mlp_bias").get<bool>() && !c.at("norm_bias").get<bool>() &&
              c.at("hidden_activation") == "gelu",
          "decision_architecture");
    m.hidden = integer(c, "hidden_size");
    m.intermediate = integer(c, "intermediate_size");
    m.layers = integer(c, "num_hidden_layers");
    m.heads = integer(c, "num_attention_heads");
    m.vocab = integer(c, "vocab_size");
    m.head_layers = integer(m.config, "head_layers");
    m.max_len = integer(m.config, "max_len");
    m.head_len = integer(m.config, "head_max_len");
    m.window = integer(c, "local_attention") / 2;
    m.eps = c.at("norm_eps");
    check(m.hidden >= 8 && m.hidden <= 1024 && m.intermediate > 0 && m.intermediate <= 4096 &&
              m.layers > 0 && m.layers <= 28 && m.head_layers >= 0 && m.head_layers <= 2 &&
              m.heads > 0 && m.heads <= 32 && m.hidden % m.heads == 0 &&
              (m.hidden / m.heads) % 2 == 0 && m.hidden % std::max(1, m.hidden / 64) == 0 &&
              m.vocab > 0 && m.vocab <= 65536 && m.max_len > 0 && m.max_len <= 512 &&
              m.head_len >= 16 && m.head_len <= 192 && m.window > 0 && m.window <= 512 &&
              m.eps > 0 && m.eps < 1,
          "decision_dimensions");
    check(c.at("layer_types").size() == std::size_t(m.layers), "decision_layer_types");
    for (auto &t : c.at("layer_types"))
        check(t == "full_attention" || t == "sliding_attention", "decision_layer_type");
    auto rope = c.at("rope_parameters");
    for (auto name : {"full_attention", "sliding_attention"})
        check(rope.at(name).at("rope_type") == "default", "decision_rope");
    m.global_theta = rope.at("full_attention").at("rope_theta");
    m.local_theta = rope.at("sliding_attention").at("rope_theta");
    check(m.global_theta >= 1 && m.local_theta >= 1 && m.global_theta <= 1e8 &&
              m.local_theta <= 1e8,
          "decision_rope");
    m.tokenizer =
        std::make_unique<Tokenizer>(read_json(root + "/tokenizer/tokenizer.json", 8 * MiB));
    cancel_check(cancel);
    auto metadata = std::make_shared<checkpoint::MemoryBudget>(16 * MiB);
    auto source = std::filesystem::canonical(std::filesystem::path(root) / "model.safetensors");
    auto shard = std::make_shared<checkpoint::Shard>(source.parent_path().c_str(),
                                                     source.filename().string(), metadata,
                                                     checkpoint::Limits{MiB, 1024, 512});
    serving::WeightBacking backing(resources_, 0, 1024 * MiB, 65536);
    std::vector<std::pair<std::string, std::vector<std::uint64_t>>> specs;
    auto spec = [&](std::string name, std::initializer_list<std::uint64_t> shape) {
        specs.emplace_back(std::move(name), shape);
    };
    auto h = std::uint64_t(m.hidden), i = std::uint64_t(m.intermediate);
    spec("encoder.embeddings.tok_embeddings.weight", {std::uint64_t(m.vocab), h});
    spec("encoder.embeddings.norm.weight", {h});
    spec("encoder.final_norm.weight", {h});
    spec("type_emb.weight", {3, h});
    for (int n = 0; n < m.layers; ++n) {
        auto p = "encoder.layers." + std::to_string(n);
        if (n)
            spec(p + ".attn_norm.weight", {h});
        spec(p + ".mlp_norm.weight", {h});
        spec(p + ".attn.Wqkv.weight", {3 * h, h});
        spec(p + ".attn.Wo.weight", {h, h});
        spec(p + ".mlp.Wi.weight", {2 * i, h});
        spec(p + ".mlp.Wo.weight", {h, i});
    }
    for (int n = 0; n < m.head_layers; ++n) {
        auto p = "head.layers." + std::to_string(n);
        for (auto norm : {".norm1", ".norm2"}) {
            spec(p + norm + ".weight", {h});
            spec(p + norm + ".bias", {h});
        }
        spec(p + ".self_attn.in_proj_weight", {3 * h, h});
        spec(p + ".self_attn.in_proj_bias", {3 * h});
        spec(p + ".self_attn.out_proj.weight", {h, h});
        spec(p + ".self_attn.out_proj.bias", {h});
        spec(p + ".linear1.weight", {4 * h, h});
        spec(p + ".linear1.bias", {4 * h});
        spec(p + ".linear2.weight", {h, 4 * h});
        spec(p + ".linear2.bias", {h});
    }
    spec("scorer.0.weight", {h});
    spec("scorer.0.bias", {h});
    spec("scorer.1.weight", {h, h});
    spec("scorer.1.bias", {h});
    spec("scorer.3.weight", {1, h});
    spec("scorer.3.bias", {1});
    spec("act_head.0.weight", {256, h + 4});
    spec("act_head.0.bias", {256});
    spec("act_head.2.weight", {2, 256});
    spec("act_head.2.bias", {2});
    Bytes bytes = 0;
    for (auto &[name, shape] : specs) {
        auto t = shard->tensor(name);
        check(t.dtype == checkpoint::Dtype::fp16 && t.rank == shape.size() &&
                  std::equal(shape.begin(), shape.end(), t.shape.begin()),
              "decision_weight_shape");
        bytes += t.bytes * 2;
    }
    check(bytes <= 2 * 1024 * MiB, "decision_weight_budget");
    if (admission)
        resources_->resize_loading(admission->h, host(resources_, metadata_bytes + bytes));
    std::array<std::uint8_t, 65536> buffer{};
    for (auto &[name, shape] : specs) {
        if (observer)
            observer("weights");
        cancel_check(cancel);
        auto t = shard->tensor(name);
        Tensor value{shape, Matrix(t.bytes / 2)};
        for (std::size_t offset = 0; offset < t.bytes;) {
            cancel_check(cancel);
            auto size = std::min<std::size_t>(buffer.size(), t.bytes - offset);
            backing.read_through(name, Workload::decision, shard, name, offset,
                                 {buffer.data(), size}, &cancel);
            for (std::size_t b = 0; b < size; b += 2)
                value.values[(offset + b) / 2] =
                    half(std::uint16_t(buffer[b]) | (std::uint16_t(buffer[b + 1]) << 8));
            offset += size;
        }
        m.weights.emplace(name, std::move(value));
    }
    shard->check_unchanged();
    cancel_check(cancel);
    if (admission) {
        resources_->loaded(admission->h);
        resident_ = admission->h;
        admission->h = 0;
    } else {
        resident_ = envelope;
        borrowed_ = true;
    }
    impl_ = std::move(next);
}
std::string Executor::execute(std::string_view request, const std::atomic_bool &cancel,
                              const Observer &observer) {
    check(loaded(), "decision_not_loaded");
    check(request.size() <= 65536, "decision_request_size");
    cancel_check(cancel);
    Pin pin(resources_, resident_);
    std::unique_ptr<Reservation> scratch;
    if (!borrowed_)
        scratch = std::make_unique<Reservation>(resources_, scratch_bytes);
    auto body = parse_json(request);
    check(body.is_object() && body.size() == 2 && body.contains("state") &&
              body.contains("questions"),
          "decision_request_fields");
    const auto state = string_field(body, "state");
    const auto &questions = body.at("questions");
    check(questions.is_array() && !questions.empty() && questions.size() <= 8,
          "decision_questions");
    auto &m = *impl_;
    auto state_ids = m.tokenizer->encode(state, cancel);
    Json answers = Json::array();
    std::set<std::string> question_keys;
    for (auto &q : questions) {
        cancel_check(cancel);
        auto key = string_field(q, "key"), kind = string_field(q, "type"),
             ins = string_field(q, "instructions");
        check(question_keys.insert(key).second, "decision_duplicate_question");
        int type = kind == "Choice" ? 0 : kind == "Score" ? 1 : kind == "Noul" ? 2 : -1;
        check(type >= 0, "decision_question_type");
        std::vector<std::string> labels, options;
        if (type == 0) {
            check(q.size() == 4 && q.at("options").is_array() && !q.at("options").empty() &&
                      q.at("options").size() <= 64,
                  "decision_choice_options");
            std::set<std::string> keys;
            for (auto &opt : q.at("options")) {
                auto label = string_field(opt, "key"),
                     description = string_field(opt, "description", true);
                check(opt.size() == 2 && keys.insert(label).second, "decision_duplicate_option");
                labels.push_back(label);
                options.push_back(description.empty() ? label : label + ": " + description);
            }
        } else if (type == 1) {
            check(q.size() == 4 && q.at("levels").is_array() && q.at("levels").size() >= 1 &&
                      q.at("levels").size() <= 64,
                  "decision_score_levels");
            for (auto &level : q.at("levels")) {
                check(level.is_string() && !level.get<std::string>().empty(),
                      "decision_score_level");
                auto label = std::to_string(labels.size());
                labels.push_back(label);
                options.push_back("level " + label + ": " + level.get<std::string>());
            }
        } else {
            for (auto it = q.begin(); it != q.end(); ++it)
                check(it.key() == "key" || it.key() == "type" || it.key() == "instructions" ||
                          it.key() == "trueWhen" || it.key() == "falseWhen",
                      "decision_noul_fields");
            auto no = q.value("falseWhen", std::string{}), yes = q.value("trueWhen", std::string{});
            labels = {"false", "true"};
            options = {"false: " + (no.empty() ? "no, the statement does not hold" : no),
                       "true: " + (yes.empty() ? "yes, the statement holds" : yes)};
        }
        auto prefix = type == 0 ? "choice" : type == 1 ? "score" : "noul";
        auto head = m.tokenizer->encode(std::string(prefix) + " question: " + ins, cancel);
        std::vector<std::vector<int>> encoded;
        std::size_t option_size = 0;
        for (auto &option : options) {
            auto ids = m.tokenizer->encode(" " + option, cancel);
            check(ids.size() <= 48, "decision_option_token_budget");
            ids.insert(ids.begin(), m.tokenizer->mask);
            option_size += ids.size();
            encoded.push_back(std::move(ids));
        }
        int budget = m.head_len - int(option_size);
        if (budget < 16) {
            int per = std::max(4, (m.head_len - 16) / int(options.size()));
            for (auto &ids : encoded)
                check(int(ids.size()) <= per, "decision_head_token_budget");
        }
        check(int(head.size()) <= std::max(8, budget), "decision_instruction_token_budget");
        std::vector<int> ids{m.tokenizer->cls};
        ids.insert(ids.end(), head.begin(), head.end());
        ids.push_back(m.tokenizer->sep);
        std::vector<int> markers;
        for (auto &option : encoded) {
            markers.push_back(int(ids.size()));
            ids.insert(ids.end(), option.begin(), option.end());
        }
        ids.push_back(m.tokenizer->sep);
        ids.insert(ids.end(), state_ids.begin(), state_ids.end());
        ids.push_back(m.tokenizer->sep);
        check(int(ids.size()) <= m.max_len, "decision_state_token_budget");
        if (observer)
            observer("tokenized");
        cancel_check(cancel);
        auto x = m.forward(ids, type, cancel, observer);
        Matrix selected;
        for (int marker : markers)
            selected.insert(selected.end(), x.begin() + marker * m.hidden,
                            x.begin() + (marker + 1) * m.hidden);
        norm(selected, m.hidden, m.w("scorer.0.weight"), &m.w("scorer.0.bias"), 1e-5f);
        auto logits = linear(selected, m.hidden, m.w("scorer.1.weight"), &m.w("scorer.1.bias"));
        for (auto &v : logits)
            v = gelu(v);
        logits = linear(logits, m.hidden, m.w("scorer.3.weight"), &m.w("scorer.3.bias"));
        auto p = softmax(logits);
        auto sorted = p;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        float entropy = 0;
        for (float v : p)
            entropy -= v * std::log(std::max(v, 1e-9f));
        float k = float(std::max<std::size_t>(2, p.size()));
        Matrix act(x.begin(), x.begin() + m.hidden);
        act.insert(act.end(), {sorted[0], sorted[0] - (sorted.size() > 1 ? sorted[1] : 0.f),
                               entropy / std::log(k), k / 255.f});
        auto action = linear(act, m.hidden + 4, m.w("act_head.0.weight"), &m.w("act_head.0.bias"));
        for (float &v : action)
            v = gelu(v);
        action = softmax(linear(action, 256, m.w("act_head.2.weight"), &m.w("act_head.2.bias")));
        check(action.size() == 2, "decision_action_output");
        auto bucket = std::string(prefix) + ":" +
                      (p.size() <= 2    ? "2"
                       : p.size() <= 5  ? "3-5"
                       : p.size() <= 10 ? "6-10"
                                        : "11+");
        float temp = temperature(m.config.at("temperature").at(type));
        if (m.config.contains("temperature_by_options") &&
            m.config.at("temperature_by_options").contains(bucket))
            temp = temperature(m.config.at("temperature_by_options").at(bucket));
        for (auto &v : logits)
            v /= temp;
        p = softmax(logits);
        auto best = std::size_t(std::max_element(p.begin(), p.end()) - p.begin());
        Json probabilities = Json::object();
        for (std::size_t n = 0; n < p.size(); ++n)
            probabilities[labels[n]] = rounded(p[n]);
        Json value;
        if (type == 0)
            value = labels[best];
        else if (type == 1) {
            double score = 0;
            for (std::size_t n = 0; n < p.size(); ++n)
                score += double(n) * double(p[n]);
            value = rounded(score);
        } else
            value = rounded(p[1]);
        answers.push_back({{"key", key},
                           {"type", kind},
                           {"value", value},
                           {"confidence", rounded(p[best])},
                           {"probabilities", type == 2 ? Json(nullptr) : probabilities}});
    }
    cancel_check(cancel);
    if (observer)
        observer("publication");
    cancel_check(cancel);
    auto result = Json{{"answers", answers}}.dump();
    check(result.size() <= 65536, "decision_output_size");
    return result;
}
} // namespace kadan::decision
