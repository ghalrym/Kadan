#include "stack_checks.hpp"
#include "kadan/cuda_stack.hpp"
#include <cuda_runtime_api.h>
#include <charconv>
#include <iostream>
#include <string_view>
namespace {
void cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int integer(const char*s){std::string_view v(s);int n=-1;auto[end,error]=std::from_chars(v.data(),v.data()+v.size(),n);stack_test::check(error==std::errc{}&&end==v.data()+v.size(),"argument");return n;}
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}stack_test::check(caught,"expected_rejection");}
}
int main(int argc,char**argv){
    if(argc!=6||std::string_view(argv[1])!="--allow-gpu-validation"||std::string_view(argv[2])!="--device"||std::string_view(argv[4])!="--stage")return 2;
    try{
        using namespace stack_test;const int device=integer(argv[3]),stage=integer(argv[5]);check(device>=0&&device<64&&stage>=1&&stage<=3,"argument");cuda(cudaSetDevice(device));
        for(int variant=0;variant<(stage==3?2:1);++variant){
            StackFixture fixture;auto c=fixture.config();auto p=kadan::stack::plan(c);
            if(stage==3&&variant==0)for(std::size_t e=0;e<4;++e){fixture.layer[3].moe.globals[3*e]=1e20f;fixture.layer[3].moe.globals[3*e+1]=1e20f;}
            if(stage==3&&variant==1){fixture.packed.fill(0);c.eos=0;}
            kadan::Footprint cap(device+2,0);cap[0]=524288;cap[device+1]=131072;auto resources=std::make_shared<kadan::Resources>(cap);
            kadan::cuda::Stack model(c,fixture.weights(),device,resources);
            check(resources->snapshot().used[0]==model.host_metadata_bytes()&&resources->snapshot().used[device+1]==p.device_bytes&&resources->snapshot().residents==1,"one_envelope");
            if(stage<=2){for(int replay=0;replay<(stage==1?1:2);++replay){unsigned next=0;for(std::size_t t=0;t<(stage==1?1:5);++t){
                unsigned input=t<2?stack_golden::input_ids[t]:next;check(input==stack_golden::input_ids[t],"feedback");auto selected=model.step(input,t!=0);next=selected.token;check(next==stack_golden::selected[t]&&!selected.eos&&model.valid()&&model.tokens()==t+1,"publication_selection");verify(model,p,t);
            }model.reset();zero(model,p);}}
            else if(variant==0){
                auto result=kadan::stack::Selection{99,false};bool failed=false;try{result=model.step(2);}catch(const std::overflow_error&){failed=true;}
                check(failed&&result.token==99&&!model.valid()&&model.tokens()==0,"late_failure_publication");rejects([&]{model.step(2);});std::array<float,16>a{},b{};rejects([&]{model.read_output(a,b);});model.reset();zero(model,p);
                std::atomic_bool cancelled=true;rejects([&]{model.step(2,true,&cancelled);});check(!model.valid()&&model.tokens()==0,"cancel_publication");model.reset();zero(model,p);
            }else{auto selected=model.step(2);check(selected.token==0&&selected.eos&&model.finished()&&model.tokens()==1,"greedy_tie_eos");rejects([&]{model.step(0);});check(model.tokens()==1,"eos_no_progress");model.reset();zero(model,p);}
            std::array<float,16>a{},b{};rejects([&]{model.read_output(a,b);});model.close();model.close();check(!model.valid()&&resources->snapshot().used[0]==0&&resources->snapshot().used[device+1]==0&&resources->snapshot().residents==0,"cleanup");
            std::cout<<"Mixed stack stage "<<stage<<" variant "<<variant<<" passed; "<<p.device_bytes<<" requested device bytes, "<<model.host_metadata_bytes()<<" owner RAM; zero ledger. Synthetic correctness only.\n";
        }
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
