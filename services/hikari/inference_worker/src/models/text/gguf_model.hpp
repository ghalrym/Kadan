#pragma once
// Shared pieces of the native GGUF text models: tensor binding, bounded device
// projections, F32 layer math and the per-architecture model interface. Layer
// math follows llama.cpp's GGUF graphs and uses F32 activation arithmetic.
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gguf_placement.hpp"
#include "kadan/dense_compute.hpp"
#include "kadan/gguf_checkpoint.hpp"

namespace kadan::text {
using Vec = std::vector<float>;
using Progress = std::function<void(const char*, std::size_t)>;

void need(bool condition, const char* error);

// Every tensor of the checkpoint is bound to an exact GGML shape (columns,
// rows, experts) or explicitly skipped before execution starts.
class GgufWeights {
public:
    GgufWeights(const char* root, const std::atomic_bool* cancel);

    const checkpoint::GgufCheckpoint& gguf() const {
        return *gguf_;
    }

    void set_cancel(const std::atomic_bool& cancel) {
        cancel_ = &cancel;
    }

    void check_cancel() const;

    const std::atomic_bool& cancel() const {
        return *cancel_;
    }

    // F32 parameters read whole on the host.
    void vector(const std::string& name, std::initializer_list<std::uint64_t> shape);
    // A matrix or expert stack: F32 or any gguf::Encoding.
    void matrix(
        const std::string& name,
        std::size_t rows,
        std::size_t columns,
        std::size_t experts = 1
    );
    // A table read by rows() or decode() only; never placed on a device.
    void table(
        const std::string& name,
        std::size_t rows,
        std::size_t columns,
        std::size_t experts = 1
    );
    void skip(std::string_view prefix);
    void finish();
    Vec read(const std::string& name) const;
    // Every decoded value of a small table, expert-major then row-major.
    Vec decode(const std::string& name) const;
    // Decoded whole rows of a matrix, for embedding lookups.
    void
    rows(const std::string& name, std::size_t first, std::span<float> output) const;
    // input holds whole rows of `columns`; returns rows of `rows`.
    Vec project(
        DenseCompute& compute,
        const std::string& name,
        std::span<const float> input,
        std::size_t expert = 0
    ) const;
    void check_unchanged() const;

    // Host RAM for weight slices that do not win device residency.
    void set_ram_budget(Bytes bytes) {
        ram_budget_ = bytes;
    }

    // Ages slice usage scores by processed tokens.
    void advance(std::size_t tokens) const {
        placement_.advance(tokens);
    }

    Placement::Stats placement() const {
        return placement_.stats();
    }

    std::uint64_t packed_bytes() const {
        return packed_;
    }

    // Decoded F32 bytes of every bound matrix, for device weight planning.
    std::uint64_t execution_bytes() const {
        return execution_;
    }

private:
    struct Matrix {
        std::size_t rows, columns, experts;
    };

    std::unique_ptr<checkpoint::GgufCheckpoint> gguf_;
    std::map<std::string, Matrix, std::less<>> matrices_;
    std::map<std::string, std::size_t, std::less<>> vectors_;
    std::set<std::string, std::less<>> bound_;
    std::vector<std::string> skipped_;
    std::uint64_t packed_ = 0, execution_ = 0;
    std::atomic_bool idle_cancel_{false};
    const std::atomic_bool* cancel_ = &idle_cancel_;
    // Projections run on one model thread; device reader callbacks only read
    // host slices, which change between projections.
    Bytes ram_budget_ = 0;
    mutable bool budgeted_ = false;
    mutable Placement placement_;
    mutable std::map<SliceKey, std::vector<std::byte>> ram_;

    void place(DenseCompute&, const SliceKey&, Bytes bytes, Placement::Decision&) const;

    void bind(const std::string& name);
};

float sigmoid(float x);
float silu(float x);
// ggml_softplus: linear above 20.
float softplus(float x);
// RMSNorm over consecutive groups of weight.size(), each scaled by weight.
void rms_norm(std::span<float> x, std::span<const float> weight, float eps);
// Unweighted RMSNorm over groups of `width`.
void rms_norm(std::span<float> x, std::size_t width, float eps);
// x / sqrt(sum(x^2) + eps) over groups of `width`, as build_gdn_l2_norm.
void l2_norm(std::span<float> x, std::size_t width, float eps);
// NEOX rotation of the first `rotary` values of each `width` head. Text-only
// interleaved M-RoPE reduces to this because every section shares the position.
void rope(
    std::span<float> x,
    std::size_t width,
    std::size_t rotary,
    std::size_t position,
    float base
);

// Grouped-query attention for `queries` rows at positions offset..offset+queries-1
// over keys[0..offset+queries). Q/out use [row, head, dim]; K/V [key, kv, dim].
// `scale` multiplies dot products. Uses the device when available.
void attend(
    DenseCompute& compute,
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::size_t queries,
    std::size_t keys,
    std::size_t heads,
    std::size_t kv_heads,
    std::size_t dim,
    std::size_t offset,
    float scale,
    std::span<float> out,
    const std::atomic_bool& cancel
);

// Gated delta rule for one head and token. state holds S[key][value]; decay
// has one entry (scalar gate) or `dim` entries (per-key gate). q is
// pre-scaled by the caller; out receives sum_i S[i][j] q[i].
void delta_rule(
    std::span<float> state,
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::span<const float> decay,
    float beta,
    std::span<float> out
);

// One architecture's forward pass over whole token rows; KV and recurrent
// state live in the model and are sized for its context at construction.
class Model {
public:
    virtual ~Model() = default;
    virtual std::size_t vocabulary() const = 0;
    virtual bool eos(unsigned token) const = 0;
    virtual void reset() = 0;
    // Rows at positions position..position+tokens.size()-1. Returns the last
    // row's logits when requested, else an empty vector.
    virtual Vec forward(
        std::span<const unsigned> tokens,
        std::size_t position,
        bool logits,
        const Progress& progress
    ) = 0;
};

// Exact metadata contracts, shared by acquisition and loading.
void validate_qwen35(const checkpoint::GgufMetadata&);
void validate_qwen35(const checkpoint::GgufCheckpoint&);
void validate_qwen4exp(const checkpoint::GgufMetadata&);
void validate_qwen4exp(const checkpoint::GgufCheckpoint&);
void validate_glm(const checkpoint::GgufMetadata&);
void validate_glm(const checkpoint::GgufCheckpoint&);

// Binds every tensor; the model is constructed only after finish().
void bind_qwen35(GgufWeights&);
void bind_qwen4exp(GgufWeights&);
void bind_glm(GgufWeights&);

// Host bytes of KV/recurrent state for a context.
std::size_t qwen35_state_bytes(std::size_t context);
std::size_t qwen4exp_state_bytes(std::size_t context);
std::size_t glm_state_bytes(std::size_t context);

std::unique_ptr<Model>
qwen35_model(const GgufWeights&, DenseCompute&, std::size_t context);
std::unique_ptr<Model>
qwen4exp_model(const GgufWeights&, DenseCompute&, std::size_t context);
std::unique_ptr<Model>
glm_model(const GgufWeights&, DenseCompute&, std::size_t context);
}  // namespace kadan::text
