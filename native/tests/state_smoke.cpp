// Opt-in storage correctness only, not layer math, generation or throughput.
#include "kadan/cuda_state.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void check(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void fails(F fn,std::string_view expected) {
    bool caught=false; try { fn(); } catch (const std::invalid_argument& e) { check(e.what()==expected,"unexpected_state_error"); caught=true; }
    check(caught,"missing_state_error");
}
void put(void* device,std::size_t count,std::uint8_t value) {
    std::vector<std::uint8_t> bytes(count,value);
    check(cudaMemcpy(device,bytes.data(),count,cudaMemcpyHostToDevice)==cudaSuccess,"state_fixture_copy_failed");
}
}
int main(int argc,char** argv) {
    if (argc!=4 || std::string_view(argv[1])!="--allow-gpu-validation" || std::string_view(argv[2])!="--device") {
        std::cerr<<"Explicit reviewed GPU authorization required: kadan-state-smoke --allow-gpu-validation --device ORDINAL\n"; return 2;
    }
    try {
        int device=-1; const std::string_view value(argv[3]);
        const auto [end,error]=std::from_chars(value.data(),value.data()+value.size(),device);
        check(error==std::errc{} && end==value.data()+value.size() && device>=0 && device<64,"invalid_device");
        check(cudaSetDevice(device)==cudaSuccess,"state_fixture_device_failed");
        kadan::checkpoint::TextArchitecture a{};
        a.layers=2; a.max_context=4; a.attention_heads=2; a.kv_heads=1; a.head_dim=8;
        a.key_heads=1; a.value_heads=2; a.key_dim=8; a.value_dim=8; a.conv_kernel=3;
        a.layer_types[0]=kadan::checkpoint::LayerKind::linear_attention;
        a.layer_types[1]=kadan::checkpoint::LayerKind::full_attention;
        kadan::Footprint capacity(device+2,0); capacity[device+1]=65536;
        auto resources=std::make_shared<kadan::Resources>(capacity);
        kadan::cuda::SequenceState state(a,3,device,resources);
        check(state.plan().device_bytes[device]==1280 && resources->snapshot().used[device+1]==1280,"state_bad_admission");
        std::vector<std::uint8_t> expected(1280),actual(1280);
        state.read_bytes(0,actual); check(actual==expected,"state_not_initially_zero");
        for (std::size_t token=0;token<2;++token) {
            const auto step=state.begin(); const auto linear=state.layer(step,0),full=state.layer(step,1);
            check(linear.conv_bytes==192 && linear.recurrent_bytes==512 && linear.key_history==nullptr,"state_linear_layout");
            check(full.history_tokens==token+1 && full.kv_token_bytes==16 && full.recurrent==nullptr,"state_full_layout");
            put(linear.convolution,192,0x20+token); put(linear.recurrent,512,0x30+token);
            put(full.key_current,16,0x40+token); put(full.value_current,16,0x50+token);
            // Independent tiny-layout golden offsets; padding and old KV slots stay intact.
            std::fill_n(expected.begin(),192,0x20+token); std::fill_n(expected.begin()+256,512,0x30+token);
            std::fill_n(expected.begin()+768+16*token,16,0x40+token);
            std::fill_n(expected.begin()+1024+16*token,16,0x50+token);
            state.written(step,0); state.written(step,1); state.commit(step);
            state.read_bytes(0,actual); check(actual==expected && state.committed_tokens()==token+1,"state_persistence_failed");
        }
        const auto aborted=state.begin(); const auto full=state.layer(aborted,1);
        put(full.key_current,16,0x60); state.written(aborted,1);
        fails([&]{ state.commit(aborted); },"state_incomplete_step"); state.abort(aborted);
        fails([&]{ state.begin(); },"state_unavailable");
        fails([&]{ state.read_bytes(0,actual); },"state_read_unavailable");
        state.reset(); state.read_bytes(0,actual);
        check(std::all_of(actual.begin(),actual.end(),[](auto x){return x==0;}) && state.committed_tokens()==0,"state_reset_failed");
        fails([&]{ state.layer(aborted,0); },"stale_state_step");
        state.close(); state.close();
        check(resources->snapshot().residents==0 && resources->snapshot().used[device+1]==0,"state_cleanup_failed");
        std::cout<<"Passed one 1280-byte hybrid state arena: zero, two commits, abort invalidation, reset, stale handle and cleanup. No model/layer math or throughput test.\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
