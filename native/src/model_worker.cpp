#include "kadan/model_worker.hpp"
#include "kadan/model.hpp"
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string_view>
#ifdef KADAN_WORKER_CUDA
#include "kadan/cuda_model.hpp"
#include <cuda_runtime_api.h>
#endif
namespace {
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
struct Plan {std::size_t host,arena,vocab,capacity,staging;};
Plan plan(const char* root,std::size_t capacity,std::size_t metadata){
    require(capacity>0&&capacity<=kadan::serving::max_capacity,"capacity_range");
    require(metadata>0&&metadata<=kadan::serving::metadata_bytes,"metadata_range");
    auto budget=std::make_shared<kadan::checkpoint::MemoryBudget>(metadata);
    kadan::checkpoint::ModelManifest manifest(root,budget);
    auto generation=kadan::model::read_generation(root,manifest.architecture().vocab,budget);
    kadan::model::Layout layout(manifest,capacity,generation,budget);
    require(layout.minimum_staging_bytes()<=kadan::serving::staging_bytes,"staging_range");
    return {metadata+kadan::serving::staging_bytes+kadan::serving::control_bytes,layout.device_bytes(),manifest.architecture().vocab,capacity,layout.minimum_staging_bytes()};
}
#ifdef KADAN_WORKER_CUDA
class CudaEngine final:public kadan::serving::Engine {
    std::shared_ptr<kadan::Resources> resources_;
    std::unique_ptr<kadan::cuda::Model> model_;
    kadan::serving::Info info_;
public:
    CudaEngine(const char* root,std::size_t device,Plan p,std::size_t host,std::size_t gpu,std::size_t headroom):info_{p.vocab,p.capacity,p.arena,p.host}{
        require(device<kadan::Resources::max_devices,"device_range");
        require(headroom>=512*1024*1024,"headroom_range");
        require(host>=p.host&&gpu>=p.arena&&headroom<=gpu-p.arena,"envelope_exhausted");
        kadan::Footprint capacity(device+2);capacity[0]=host;capacity[device+1]=gpu;
        resources_=std::make_shared<kadan::Resources>(capacity);
        kadan::cuda::ModelOptions options;options.capacity=p.capacity;options.device_headroom=headroom;
        auto status=cudaSetDevice(int(device));require(status==cudaSuccess,"cuda_set_device");
        model_=std::make_unique<kadan::cuda::Model>(root,options,int(device),resources_);
        require(model_->vocabulary()==p.vocab&&model_->device_bytes()==p.arena,"preflight_changed");
    }
    kadan::serving::Info info()const override{return info_;}
    void reset()override{model_->reset();}
    kadan::serving::Token step(unsigned token,bool stop)override{auto selected=model_->step(token,stop);return {selected.token,selected.eos,model_->tokens()};}
    void close()override{model_->close();model_.reset();auto state=resources_->snapshot();require(state.residents==0,"reservation_leak");for(auto bytes:state.used)require(bytes==0,"reservation_leak");}
};
#endif
}
int main(int argc,char** argv){
    try{
        require(std::signal(SIGPIPE,SIG_IGN)!=SIG_ERR,"sigpipe_setup");
        using kadan::serving::number;
        if(argc==5&&std::string_view(argv[1])=="--plan"){
            auto p=plan(argv[2],number(argv[3]),number(argv[4]));
            std::cout<<"plan 1 "<<p.host<<' '<<p.arena<<' '<<p.vocab<<' '<<p.capacity<<' '<<p.staging<<'\n';std::cout.flush();require(bool(std::cout),"output_failed");return 0;
        }
        require(argc==8&&std::string_view(argv[1])=="--serve","usage");
#ifdef KADAN_WORKER_CUDA
        const auto device=number(argv[3]),capacity=number(argv[4]),host=number(argv[5]),gpu=number(argv[6]),headroom=number(argv[7]);
        auto p=plan(argv[2],capacity,kadan::serving::metadata_bytes);
        CudaEngine engine(argv[2],device,p,host,gpu,headroom);
        kadan::serving::session(engine,std::cin,std::cout);return 0;
#else
        throw std::runtime_error("cuda_build_required");
#endif
    }catch(const std::exception& error){std::cerr<<"model_worker: "<<error.what()<<'\n';std::cout<<"error terminal\n";std::cout.flush();return 1;}
}
