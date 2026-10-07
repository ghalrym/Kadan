#include "kadan/sequence_state.hpp"
#include <atomic>
#include <limits>
#include <stdexcept>

namespace kadan {
namespace {
void require(bool ok,const char* error) { if (!ok) throw std::invalid_argument(error); }
std::size_t add(std::size_t a,std::size_t b) {
    require(b<=std::numeric_limits<std::size_t>::max()-a,"state_size_overflow"); return a+b;
}
std::size_t mul(std::size_t a,std::size_t b) {
    require(!b || a<=std::numeric_limits<std::size_t>::max()/b,"state_size_overflow"); return a*b;
}
std::size_t align(std::size_t n) { return add(n,255)&~std::size_t{255}; }
// One monotonically assigned identity per cursor, including replacement objects at
// reused addresses. Saturate rather than wrap; relaxed ordering only assigns IDs.
std::uint64_t new_owner() {
    static std::atomic<std::uint64_t> last{0};
    auto value=last.load(std::memory_order_relaxed);
    for (;;) {
        require(value!=std::numeric_limits<std::uint64_t>::max(),"state_owner_exhausted");
        if (last.compare_exchange_weak(value,value+1,std::memory_order_relaxed)) return value+1;
    }
}
void dimension(std::size_t n,std::size_t limit) { require(n>0 && n<=limit,"state_architecture_dimension"); }
}
SequenceStatePlan plan_sequence_state(const checkpoint::TextArchitecture& a,std::size_t capacity,
    std::span<const std::size_t> placements,std::span<const std::size_t> budgets) {
    dimension(a.layers,256); dimension(a.max_context,2147483647);
    require(capacity>0 && capacity<=a.max_context,"state_context_capacity");
    require(placements.size()==a.layers && !budgets.empty() && budgets.size()<=64,"state_placement_shape");
    dimension(a.attention_heads,1024); dimension(a.kv_heads,1024); dimension(a.head_dim,4096);
    dimension(a.key_heads,1024); dimension(a.value_heads,1024); dimension(a.key_dim,4096);
    dimension(a.value_dim,4096); dimension(a.conv_kernel,1024);
    require(a.attention_heads%a.kv_heads==0 && a.value_heads%a.key_heads==0,"state_head_grouping");
    const auto channels=add(mul(mul(a.key_heads,a.key_dim),2),mul(a.value_heads,a.value_dim));
    const auto conv=mul(mul(channels,a.conv_kernel),2);
    const auto recurrent=mul(mul(mul(a.value_heads,a.key_dim),a.value_dim),4);
    const auto kv_token=mul(mul(a.kv_heads,a.head_dim),2);
    SequenceStatePlan plan{}; plan.layers=a.layers; plan.token_capacity=capacity; plan.devices=budgets.size();
    for (std::size_t i=0;i<a.layers;++i) {
        const auto device=placements[i]; require(device<budgets.size(),"state_device");
        auto& layer=plan.layer[i]; layer.device=device; layer.kind=a.layer_types[i];
        auto& end=plan.device_bytes[device];
        const auto region=[&](std::size_t bytes) { const auto offset=end; end=align(add(end,bytes)); return offset; };
        if (layer.kind==checkpoint::LayerKind::full_attention) {
            layer.kv_token_bytes=kv_token;
            layer.key_offset=region(mul(capacity,kv_token));
            layer.value_offset=region(mul(capacity,kv_token));
        } else {
            require(layer.kind==checkpoint::LayerKind::linear_attention,"state_layer_kind");
            layer.conv_bytes=conv; layer.recurrent_bytes=recurrent;
            layer.conv_offset=region(conv); layer.recurrent_offset=region(recurrent);
        }
        require(end<=budgets[device],"state_device_budget");
    }
    return plan;
}
StateCursor::StateCursor(std::size_t layers,std::size_t capacity):layers_(layers),capacity_(capacity),owner_(new_owner()) {
    require(layers>0 && layers<=256 && capacity>0,"state_cursor_shape");
}
StateStep StateCursor::begin() {
    require(valid() && !active_,"state_unavailable"); require(tokens_<capacity_,"state_context_full");
    require(next_!=std::numeric_limits<std::uint64_t>::max(),"state_step_exhausted");
    written_.reset(); active_=++next_; return StateStep(owner_,active_);
}
void StateCursor::check_step(StateStep step) const { require(valid() && active_ && step.owner_==owner_ && active_==step.generation_,"stale_state_step"); }
void StateCursor::written(StateStep step,std::size_t layer) {
    check_step(step); require(layer<layers_,"state_layer_index"); require(!written_[layer],"state_layer_already_written"); written_.set(layer);
}
void StateCursor::ready_to_commit(StateStep step) const { check_step(step); require(written_.count()==layers_,"state_incomplete_step"); }
void StateCursor::commit(StateStep step) { ready_to_commit(step); ++tokens_; active_=0; }
void StateCursor::abort(StateStep step) { check_step(step); invalidate(); }
void StateCursor::reset() { require(!closed_ && !active_,"state_reset_unavailable"); tokens_=0; written_.reset(); valid_=true; }
void StateCursor::invalidate() noexcept { valid_=false; active_=0; written_.reset(); }
void StateCursor::close() noexcept { invalidate(); closed_=true; }
} // namespace kadan
