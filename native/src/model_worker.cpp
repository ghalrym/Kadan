#include "kadan/model_worker.hpp"
#include "kadan/model.hpp"
#include "kadan/weight_backing.hpp"
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
struct Plan {std::size_t host,arena,vocab,capacity,staging,packed,tensors;};
Plan plan(const char* root,std::size_t capacity,std::size_t metadata){
    require(capacity>0&&capacity<=kadan::serving::max_capacity,"capacity_range");
    require(metadata>0&&metadata<=kadan::serving::metadata_bytes,"metadata_range");
    auto budget=std::make_shared<kadan::checkpoint::MemoryBudget>(metadata);
    kadan::checkpoint::ModelManifest manifest(root,budget);
    auto generation=kadan::model::read_generation(root,manifest.architecture().vocab,budget);
    kadan::model::Layout layout(manifest,capacity,generation,budget);
    require(layout.minimum_staging_bytes()<=kadan::serving::staging_bytes,"staging_range");
    std::size_t packed=0;
    for(const auto& item:manifest.items()) { require(item.payload_bytes<=SIZE_MAX-packed,"packed_overflow"); packed+=item.payload_bytes; }
    return {metadata+kadan::serving::staging_bytes+kadan::serving::control_bytes,layout.device_bytes(),manifest.architecture().vocab,capacity,layout.minimum_staging_bytes(),packed,manifest.tensor_count()-manifest.excluded_tensor_count()};
}
#ifdef KADAN_WORKER_CUDA
class CudaEngine final:public kadan::serving::ResidentEngine {
    std::shared_ptr<kadan::Resources> resources_;
    std::unique_ptr<kadan::cuda::Model> model_;
    kadan::serving::Info info_;
public:
    CudaEngine(const char* root,std::size_t device,Plan p,std::size_t host,std::size_t gpu,std::size_t headroom,bool split=false,std::size_t cache=0,std::size_t cold=0):info_{p.vocab,p.capacity,p.arena,p.host}{
        require(device<kadan::Resources::max_devices,"device_range");
        require(headroom>=512*1024*1024,"headroom_range");
        require(host>=p.host&&gpu>=p.arena&&headroom<=gpu-p.arena,"envelope_exhausted");
        kadan::Footprint capacity(device+2);capacity[0]=host;capacity[device+1]=gpu;
        resources_=std::make_shared<kadan::Resources>(capacity);
        kadan::cuda::ModelOptions options;options.capacity=p.capacity;options.device_headroom=headroom;options.split_residency=split;options.weight_ram_bytes=cache;options.weight_cold_bytes=cold;
        auto status=cudaSetDevice(int(device));require(status==cudaSuccess,"cuda_set_device");
        model_=std::make_unique<kadan::cuda::Model>(root,options,int(device),resources_);
        require(model_->vocabulary()==p.vocab&&model_->device_bytes()==p.arena,"preflight_changed");
        if(split)model_->end_request();
    }
    std::shared_ptr<kadan::Resources> resources()const{return resources_;}
    void begin_request()override{model_->begin_request();}
    void end_request()override{model_->end_request();}
    void park()override{model_->park();}
    kadan::serving::WeightBacking::Stats cache_stats()const override{return model_->cache_stats();}
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
        if(argc==5&&(std::string_view(argv[1])=="--plan"||std::string_view(argv[1])=="--plan-resident")){
            const bool resident=std::string_view(argv[1])=="--plan-resident";
            auto p=plan(argv[2],number(argv[3]),number(argv[4]));
            if(resident)p.host+=kadan::serving::WeightBacking::model_control_bytes;
            std::cout<<"plan "<<(resident?3:1)<<' '<<p.host<<' '<<p.arena<<' '<<p.vocab<<' '<<p.capacity<<' '<<p.staging;
            if(resident)std::cout<<' '<<p.packed<<' '<<p.tensors;
            std::cout<<'\n';std::cout.flush();require(bool(std::cout),"output_failed");return 0;
        }
        const bool resident=argc==10&&std::string_view(argv[1])=="--serve-resident";
        if(resident)require(std::signal(SIGTERM,SIG_DFL)!=SIG_ERR,"sigterm_setup");
        require(resident||(argc==8&&std::string_view(argv[1])=="--serve"),"usage");
#ifdef KADAN_WORKER_CUDA
        const auto device=number(argv[3]),capacity=number(argv[4]),host=number(argv[5]),gpu=number(argv[6]),headroom=number(argv[7]);
        auto p=plan(argv[2],capacity,kadan::serving::metadata_bytes);
        if(resident)p.host+=kadan::serving::WeightBacking::model_control_bytes;
        CudaEngine engine(argv[2],device,p,host,gpu,headroom,resident,resident?number(argv[8]):0,resident?number(argv[9]):0);
        if(resident)kadan::serving::resident_session(engine,engine.resources(),kadan::serving::session_identity(),std::cin,std::cout);
        else kadan::serving::session(engine,std::cin,std::cout);
        return 0;
#else
        throw std::runtime_error("cuda_build_required");
#endif
    }catch(const std::exception& error){std::cerr<<"model_worker: "<<error.what()<<'\n';std::cout<<"error terminal\n";std::cout.flush();return 1;}
}
