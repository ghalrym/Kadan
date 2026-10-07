#include "kadan/sequence_state.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {
void check(bool ok) { if (!ok) throw std::runtime_error("state_check_failed"); }
template<class F> void fails(F fn,std::string_view expected) {
    bool caught=false; try { fn(); } catch (const std::invalid_argument& e) { check(e.what()==expected); caught=true; } check(caught);
}
kadan::checkpoint::TextArchitecture tiny() {
    kadan::checkpoint::TextArchitecture a{};
    a.layers=2; a.max_context=4; a.attention_heads=2; a.kv_heads=1; a.head_dim=8;
    a.key_heads=1; a.value_heads=2; a.key_dim=8; a.value_dim=8; a.conv_kernel=3;
    a.layer_types[0]=kadan::checkpoint::LayerKind::linear_attention;
    a.layer_types[1]=kadan::checkpoint::LayerKind::full_attention; return a;
}
void layouts() {
    auto a=tiny(); const std::array<std::size_t,2> places{0,0},budgets{1280,0};
    auto p=kadan::plan_sequence_state(a,3,places,budgets);
    check(p.layers==2 && p.token_capacity==3 && p.device_bytes[0]==1280 && p.device_bytes[1]==0);
    check(p.layer[0].conv_offset==0 && p.layer[0].conv_bytes==192);
    check(p.layer[0].recurrent_offset==256 && p.layer[0].recurrent_bytes==512);
    check(p.layer[1].key_offset==768 && p.layer[1].value_offset==1024 && p.layer[1].kv_token_bytes==16);
    check(p.layer[0].kv_token_bytes==0 && p.layer[1].conv_bytes==0);
    const std::array<std::size_t,2> split{0,1},caps{768,512};
    p=kadan::plan_sequence_state(a,3,split,caps);
    check(p.device_bytes[0]==768 && p.device_bytes[1]==512 && p.layer[1].key_offset==0 && p.layer[1].value_offset==256);
    fails([&]{ kadan::plan_sequence_state(a,0,places,budgets); },"state_context_capacity");
    fails([&]{ kadan::plan_sequence_state(a,5,places,budgets); },"state_context_capacity");
    fails([&]{ kadan::plan_sequence_state(a,3,std::span(places).first(1),budgets); },"state_placement_shape");
    fails([&]{ kadan::plan_sequence_state(a,3,places,{}); },"state_placement_shape");
    const std::array<std::size_t,2> bad_place{0,2},short_budget{1279,0};
    fails([&]{ kadan::plan_sequence_state(a,3,bad_place,budgets); },"state_device");
    fails([&]{ kadan::plan_sequence_state(a,3,places,short_budget); },"state_device_budget");
    a.layer_types[1]=static_cast<kadan::checkpoint::LayerKind>(99);
    fails([&]{ kadan::plan_sequence_state(a,3,places,budgets); },"state_layer_kind");
    a=tiny(); a.key_heads=3;
    fails([&]{ kadan::plan_sequence_state(a,3,places,budgets); },"state_head_grouping");
    a=tiny(); a.layers=257;
    fails([&]{ kadan::plan_sequence_state(a,3,places,budgets); },"state_architecture_dimension");
    a=tiny(); a.head_dim=std::numeric_limits<std::size_t>::max();
    fails([&]{ kadan::plan_sequence_state(a,3,places,budgets); },"state_architecture_dimension");
}
void actual_profile_arithmetic() {
    auto a=tiny(); a.layers=40; a.max_context=262144;
    a.attention_heads=16; a.kv_heads=2; a.head_dim=256;
    a.key_heads=16; a.value_heads=32; a.key_dim=128; a.value_dim=128; a.conv_kernel=4;
    std::array<std::size_t,40> placements{};
    for (std::size_t i=0;i<40;++i) {
        placements[i]=i<20?0:1;
        a.layer_types[i]=(i+1)%4==0?kadan::checkpoint::LayerKind::full_attention:kadan::checkpoint::LayerKind::linear_attention;
    }
    const std::array<std::size_t,2> caps{4ULL*1024*1024*1024,4ULL*1024*1024*1024};
    auto p=kadan::plan_sequence_state(a,65536,placements,caps);
    check(p.device_bytes[0]==703528960 && p.device_bytes[1]==703528960);
    p=kadan::plan_sequence_state(a,262144,placements,caps);
    check(p.device_bytes[0]==2716794880 && p.device_bytes[1]==2716794880);
    // These calls only compute byte counts, allocating no model or state buffers.
}
void lifecycle() {
    kadan::StateCursor state(2,2);
    auto step=state.begin(); check(state.active() && state.committed_tokens()==0);
    fails([&]{ state.begin(); },"state_unavailable");
    fails([&]{ state.commit(step); },"state_incomplete_step");
    fails([&]{ state.written(step,2); },"state_layer_index");
    state.written(step,1);
    fails([&]{ state.written(step,1); },"state_layer_already_written");
    state.written(step,0); state.commit(step); check(state.committed_tokens()==1 && !state.active());
    fails([&]{ state.check_step(step); },"stale_state_step");
    const auto old=step; step=state.begin(); check(step!=old);
    fails([&]{ state.written(old,0); },"stale_state_step");
    state.written(step,0); state.abort(step); check(!state.valid() && state.committed_tokens()==1);
    fails([&]{ state.begin(); },"state_unavailable");
    state.reset(); check(state.valid() && state.committed_tokens()==0);
    for (int i=0;i<2;++i) { step=state.begin(); state.written(step,0); state.written(step,1); state.commit(step); }
    fails([&]{ state.begin(); },"state_context_full");
    state.reset(); step=state.begin();
    fails([&]{ state.reset(); },"state_reset_unavailable");
    state.invalidate(); check(!state.valid() && !state.active()); state.reset();
    state.close(); state.close(); fails([&]{ state.reset(); },"state_reset_unavailable");
    fails([&]{ state.begin(); },"state_unavailable");
}
}
int main() {
    try { layouts(); actual_profile_arithmetic(); lifecycle(); }
    catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
