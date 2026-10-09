#include "kadan/cuda_model.hpp"
#include <cuda_runtime_api.h>
#include <array>
#include <bit>
#include <cmath>
#include <csignal>
#include <iostream>
#include <string_view>
#include <unistd.h>

namespace {
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void expired(int){_exit(124);}
using Model=kadan::cuda::Model;
using Record=std::array<float,16>;
struct Capture {std::array<Record,3> logits{};std::array<unsigned,3> ids{};std::array<bool,3> eos{};};
Capture capture(Model& model){
    Capture result;constexpr std::array<unsigned,3> input{2,7,11};
    for(std::size_t i=0;i<input.size();++i){auto token=model.step(input[i],false);model.read_logits(result.logits[i]);
        require(model.tokens()==i+1,"progress");result.ids[i]=token.token;result.eos[i]=token.eos;
        for(auto v:result.logits[i])require(std::isfinite(v),"nonfinite_logits");}
    return result;
}
void equal(const Capture& a,const Capture& b){
    require(a.ids==b.ids&&a.eos==b.eos,"selection_mismatch");
    for(std::size_t i=0;i<3;++i)for(std::size_t j=0;j<16;++j)
        require(std::bit_cast<unsigned>(a.logits[i][j])==std::bit_cast<unsigned>(b.logits[i][j]),"logit_mismatch");
}
std::size_t free_bytes(){std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));return free;}
void empty(const std::shared_ptr<kadan::Resources>& r){auto s=r->snapshot();require(s.residents==0,"reservation_leak");for(auto n:s.used)require(n==0,"byte_leak");}
std::size_t preflight(const char* root){
    auto metadata=std::make_shared<kadan::checkpoint::MemoryBudget>(16*1024*1024);
    kadan::checkpoint::ModelManifest manifest(root,metadata);auto a=manifest.architecture();
    require(a.layers==4&&a.hidden==16&&a.vocab==16&&a.experts==4&&manifest.checkpoint_bytes()<128*1024,"tiny_fixture_required");
    auto generation=kadan::model::read_generation(root,a.vocab,metadata);kadan::model::Layout layout(manifest,8,generation,metadata);
    require(layout.device_bytes()<128*1024,"tiny_arena_required");return layout.device_bytes();
}
}
int main(int argc,char**argv){
    // Explicit synthetic roots only. CUDA_VISIBLE_DEVICES selects one approved
    // GPU; this test will never enumerate or choose another physical device.
    if(argc!=4||std::string_view(argv[1])!="--execute"){std::cerr<<"Usage: kadan-resident-parity --execute TINY_ROOT BAD_NUMERIC_TINY_ROOT\n";return 2;}
    try{
        auto arena=preflight(argv[2]);require(preflight(argv[3])==arena,"fixture_layout_mismatch");
        require(std::signal(SIGALRM,expired)!=SIG_ERR,"watchdog");alarm(60);
        int count=0;check(cudaGetDeviceCount(&count));require(count==1,"exactly_one_visible_device_required");check(cudaSetDevice(0));
        kadan::cuda::ModelOptions options{8,16*1024*1024,1024,512*1024*1024};
        auto ledger=[&]{return std::make_shared<kadan::Resources>(kadan::Footprint{320*1024*1024,arena+options.device_headroom});};
        auto legacy_resources=ledger();Capture legacy;
        {Model model(argv[2],options,0,legacy_resources);legacy=capture(model);model.close();}empty(legacy_resources);
        const auto baseline=free_bytes();options.split_residency=true;options.weight_ram_bytes=1024*1024;options.weight_cold_bytes=128*1024;
        auto r=ledger();std::size_t retained=0,weight_device=0;
        {Model model(argv[2],options,0,r);equal(legacy,capture(model));retained=model.retained_bytes();require(retained>0,"ram_backing_empty");
            model.end_request();weight_device=r->snapshot().used[1];require(weight_device>options.device_headroom&&weight_device<arena+options.device_headroom,"state_not_released");
            model.begin_request();equal(legacy,capture(model));model.park();
            std::cout<<"PARK model_gpu_allocations=0 context_envelope="<<r->snapshot().used[1]<<" actual_gpu_free="<<free_bytes()<<'\n';
            require(r->snapshot().used[1]==options.device_headroom&&model.retained_bytes()==retained,"park_accounting");
            auto before=model.cache_stats();model.begin_request();equal(legacy,capture(model));
            auto after=model.cache_stats();require(after.source_bytes==before.source_bytes&&after.hits>before.hits,"reload_not_from_ram");
            std::atomic_bool cancelled=true;bool rejected=false;try{model.step(2,false,&cancelled);}catch(const std::exception&){rejected=true;}
            require(rejected,"cancel_not_observed");model.end_request();model.begin_request();equal(legacy,capture(model));model.close();}
        empty(r);require(free_bytes()>=baseline,"physical_gpu_leak_after_close");
        auto failed=ledger();bool rejected=false;
        try{Model bad(argv[3],options,0,failed);}catch(const std::exception&){rejected=true;}
        require(rejected,"bad_payload_accepted");empty(failed);require(free_bytes()>=baseline,"physical_gpu_leak_after_failed_load");
        auto tight=std::make_shared<kadan::Resources>(kadan::Footprint{320*1024*1024,arena+options.device_headroom-1});rejected=false;
        try{Model bad(argv[2],options,0,tight);}catch(const std::exception&){rejected=true;}
        require(rejected,"budget_overrun");empty(tight);require(free_bytes()>=baseline,"physical_gpu_leak_after_admission");alarm(0);
        std::cout<<"PASS split-vs-legacy: 4 cycles x 3 tokens x 16 logits bit-exact; selections/EOS/progress equal; arena="<<arena
                 <<" retained_ram="<<retained<<" weight_device_with_headroom="<<weight_device<<" gpu_free_baseline="<<baseline<<" gpu_free_final="<<free_bytes()
                 <<"; cancelled-step recovery, partial-load failure and budget rejection cleaned; zero final reservations\n";
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
