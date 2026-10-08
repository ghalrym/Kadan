#define KADAN_MODEL_RUNTIME
#include "stack_runtime.cpp"
#include "kadan/cuda_model.hpp"
#include "model_image.hpp"
#include <fstream>
std::size_t physical_free=SIZE_MAX;bool info_failure=false;
cudaError_t cudaMemGetInfo(std::size_t*free,std::size_t*total){if(info_failure)return cudaErrorUnknown;*free=physical_free;*total=SIZE_MAX;return cudaSuccess;}
int main(int argc,char**argv){try{
    check(argc==2||argc==3);using kadan::cuda::Model;kadan::cuda::ModelOptions o{8,16*1024*1024,64,512};
    auto metadata=std::make_shared<kadan::checkpoint::MemoryBudget>(16*1024*1024);
    if(argc==3){auto r=std::make_shared<kadan::Resources>(kadan::Footprint{Model::host_bytes(o),2*1024*1024});bool uncertain=std::string_view(argv[2])=="reject-cleanup";if(uncertain)inject(Op::free);bool quarantine=rejected([&]{Model m(argv[1],o,0,r);});check(quarantine==uncertain);if(uncertain){check(used>0&&r->snapshot().residents==1);for(auto[ptr,n]:allocations){std::free(ptr);used-=n;}allocations.clear();}else check(used==0&&r->snapshot().residents==0);return 0;}
    kadan::checkpoint::ModelManifest manifest(argv[1],metadata);auto g=kadan::model::read_generation(argv[1],16,metadata);kadan::model::Layout layout(manifest,8,g,metadata);expected_layers=int(layout.layers().size());o.staging_bytes=std::max(std::size_t(64),layout.minimum_staging_bytes());
    // Independently resolve serialized role names and compare every pointer used
    // by the streamed producer, including all routed/shared projection regions.
    ModelImage image(layout);kadan::model::load(layout,std::make_shared<kadan::checkpoint::MemoryBudget>(o.staging_bytes),image);kadan::StateCursor cursor(layout.layers().size(),8);
    for(std::size_t i=0;i<layout.layers().size();++i){auto& l=layout.layers()[i];kadan::cuda::detail::DecoderProducer p(l.config,l.plan,cursor);p.storage=image.arena.data()+l.offset;p.bind();auto base=std::string("model.language_model.layers.")+std::to_string(i);
        auto role=[&](const void* pointer,const char* suffix){check(pointer==image.arena.data()+image.binding(base+suffix).weights);};
        if(l.config.attention==kadan::decoder::Attention::linear){role(p.linear.input_norm,".input_layernorm.weight");role(p.linear.conv_weight,".linear_attn.conv1d.weight");role(p.linear.a_weight,".linear_attn.in_proj_a.weight");role(p.linear.b_weight,".linear_attn.in_proj_b.weight");role(p.linear.a_log,".linear_attn.A_log");role(p.linear.dt_bias,".linear_attn.dt_bias");role(p.linear.output_norm,".linear_attn.norm.weight");}
        else{role(p.full.input_norm,".input_layernorm.weight");role(p.full.query_norm,".self_attn.q_norm.weight");role(p.full.key_norm,".self_attn.k_norm.weight");}
        role(p.moe.router,".mlp.gate.weight");role(p.moe.shared_gate,".mlp.shared_expert_gate.weight");
        for(std::size_t e=0;e<=l.config.moe.experts;++e){auto prefix=base+(e==l.config.moe.experts?".mlp.shared_expert":".mlp.experts."+std::to_string(e));std::array<const char*,3> names{"gate_proj","up_proj","down_proj"};for(std::size_t k=0;k<3;++k){auto& b=image.binding(prefix+"."+names[k]);check(p.experts[e][k].weights==image.arena.data()+b.weights&&p.experts[e][k].scales==image.arena.data()+b.scales);}}
    }
    auto manager=[&]{return std::make_shared<kadan::Resources>(kadan::Footprint{Model::host_bytes(o),layout.device_bytes()+o.device_headroom});};
    for(int scenario=0;scenario<7;++scenario){auto r=manager();auto options=o;auto before=mallocs;
        if(scenario==0){r=std::make_shared<kadan::Resources>(kadan::Footprint{Model::host_bytes(o)-1,layout.device_bytes()+512});}
        if(scenario==1)r=std::make_shared<kadan::Resources>(kadan::Footprint{Model::host_bytes(o),layout.device_bytes()+511});
        if(scenario==2)physical_free=layout.device_bytes()+511;
        if(scenario==3)info_failure=true;
        if(scenario==4)options.metadata_bytes=64;
        if(scenario==5)options.staging_bytes=8;
        if(scenario==6)inject(Op::copy,20);
        rejected([&]{Model m(argv[1],options,0,r);});check(used==0&&r->snapshot().residents==0);if(scenario<5)check(mallocs==before);physical_free=SIZE_MAX;info_failure=false;
    }
    {auto r=manager();std::atomic_bool stop=true;auto before=mallocs;rejected([&]{Model m(argv[1],o,0,r,&stop);});check(mallocs==before&&r->snapshot().residents==0);}
    {auto r=manager();std::vector<kadan::Handle> other;for(int n=0;n<1023;++n)other.push_back(r->reserve(kadan::Workload::video,{0,0}));Model m(argv[1],o,0,r);check(r->snapshot().residents==1024&&used==layout.device_bytes());m.close();for(auto h:other)r->released(h);check(used==0);}
    for(int scenario=0;scenario<10;++scenario){auto r=manager();Model m(argv[1],o,0,r);selection_override=3;m.step(2,false);check(m.tokens()==1&&m.valid());std::atomic_bool stop=false;
        switch(scenario){case 0:late_numeric=true;break;case 1:cancellation=&stop;break;case 2:fail_final_copy=true;break;case 3:fail_final_sync=true;break;case 4:inject(Op::sync,1,2);break;case 5:inject(Op::launch);break;case 6:cancel_after_copy=&stop;break;case 7:head_numeric=true;break;case 8:selection_override=99;break;case 9:stop=true;break;}
        auto result=kadan::stack::Selection{77,false};bool quarantine=rejected([&]{result=m.step(7,false,&stop);});check(quarantine==(scenario==4));check(!m.valid()&&m.tokens()==1&&result.token==77);
        late_numeric=head_numeric=false;cancellation=cancel_after_copy=nullptr;stop=false;selection_override=3;std::array<float,16> logits{};rejected([&]{m.read_logits(logits);});auto before=launches;rejected([&]{m.step(7);});check(launches==before);
        if(scenario==0||scenario==1||scenario==6||scenario==7||scenario==8||scenario==9){m.reset();check(m.valid()&&m.tokens()==0);for(std::size_t i=0;i<layout.layers().size();++i){auto&p=layout.layers()[i].plan;std::vector<std::uint8_t>a(p.state_first_bytes),b(p.state_second_bytes);m.read_state(i,a,b);for(auto v:a)check(!v);for(auto v:b)check(!v);}m.step(2,false);}
        else rejected([&]{m.reset();});m.close();check(used==0&&r->snapshot().residents==0&&!pending);
    }
    {auto r=manager();Model m(argv[1],o,0,r);for(unsigned eos:{14,15}){selection_override=eos;check(m.step(2,false).eos&&!m.finished());check(m.step(7).eos&&m.finished());auto before=launches;rejected([&]{m.step(3);});check(launches==before&&m.tokens()==2);m.reset();}selection_override=3;
        owner_allocations::all_count=0;owner_allocations::count_all=true;for(int n=0;n<8;++n)m.step(2,false);owner_allocations::count_all=false;check(owner_allocations::all_count==0);auto before=launches;rejected([&]{m.step(2,false);});check(launches==before&&m.valid()&&m.tokens()==8);m.close();check(r->snapshot().residents==0);}
    for(auto op:{Op::free,Op::sync}){auto r=manager();std::size_t before=0;{Model m(argv[1],o,0,r);inject(op);rejected([&]{m.close();});check(r->snapshot().used[0]==Model::host_bytes(o)&&r->snapshot().used[1]==layout.device_bytes()+512);before=syncs;rejected([&]{m.close();});}check(before==syncs&&r->snapshot().residents==1);for(auto[ptr,n]:allocations){std::free(ptr);used-=n;}allocations.clear();pending=false;}
    {auto options=o;options.split_residency=true;options.weight_ram_bytes=2*1024*1024;options.weight_cold_bytes=2*1024*1024;
        auto r=std::make_shared<kadan::Resources>(kadan::Footprint{Model::host_bytes(options)+options.weight_ram_bytes,layout.device_bytes()+options.device_headroom});
        Model m(argv[1],options,0,r);check(used==layout.device_bytes()&&m.retained_bytes()>0);
        selection_override=3;m.step(2,false);m.end_request();auto weights=used;auto retained=m.retained_bytes();
        check(weights>0&&weights<layout.device_bytes()&&!m.valid());
        auto before_copies=copies,before_mallocs=mallocs;m.begin_request();check(mallocs==before_mallocs+1&&copies==before_copies&&used==layout.device_bytes());
        check(m.valid()&&m.tokens()==0);m.step(2,false);m.park();check(used==0&&m.retained_bytes()==retained&&r->snapshot().used[1]==0);
        m.begin_request();check(used==layout.device_bytes()&&m.valid());m.step(2,false);m.close();check(used==0&&r->snapshot().residents==0);
    }
    for(int scenario=0;scenario<6;++scenario){auto options=o;options.split_residency=true;options.weight_ram_bytes=65536;options.weight_cold_bytes=2*1024*1024;
        auto r=std::make_shared<kadan::Resources>(kadan::Footprint{Model::host_bytes(options)+options.weight_ram_bytes,layout.device_bytes()+options.device_headroom});
        if(scenario==4||scenario==5){if(scenario==4)options.metadata_bytes=64;else options.staging_bytes=8;rejected([&]{Model m(argv[1],options,0,r);});}
        else if(scenario==0){std::atomic_bool stop=true;rejected([&]{Model m(argv[1],options,0,r,&stop);});}
        else{Model m(argv[1],options,0,r);m.park();
            if(scenario==1)physical_free=0;if(scenario==2)inject(Op::copy);std::atomic_bool stop=scenario==3;
            rejected([&]{m.begin_request(&stop);});physical_free=SIZE_MAX;m.close();}
        check(used==0&&r->snapshot().residents==0);
    }
    for(auto op:{Op::free,Op::sync}){auto options=o;options.split_residency=true;
        auto r=std::make_shared<kadan::Resources>(kadan::Footprint{Model::host_bytes(options),layout.device_bytes()+options.device_headroom});
        {Model m(argv[1],options,0,r);inject(op);rejected([&]{m.end_request();});check(!m.valid());
            rejected([&]{m.begin_request();});rejected([&]{m.close();});}
        check(r->snapshot().used[1]==layout.device_bytes()+options.device_headroom);
        for(auto[ptr,n]:allocations){std::free(ptr);used-=n;}allocations.clear();pending=false;
    }
    check(bf16_launches>0);
    std::cout<<"Checkpoint fake runtime: "<<expected_layers<<" layers, one reservation/arena, admission/load/step failures, two EOS IDs and cleanup passed.\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
