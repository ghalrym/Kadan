#include "model_image.hpp"
#include "kadan/resources.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <source_location>
void check(bool b,std::source_location at=std::source_location::current()){if(!b)throw std::runtime_error("line_"+std::to_string(at.line()));}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed);}
template<class T,std::size_t N>void read(std::ifstream& file,std::span<T,N> v){file.read(reinterpret_cast<char*>(v.data()),v.size_bytes());check(bool(file));}
void compare(std::span<const float>a,std::span<const float>b,const char* kind,float tolerance=0){check(a.size()==b.size()&&std::isfinite(tolerance)&&tolerance>=0);for(std::size_t j=0;j<a.size();++j)if(!std::isfinite(a[j])||!std::isfinite(b[j])||std::abs(a[j]-b[j])>tolerance){std::cerr<<kind<<" index "<<j<<" actual "<<a[j]<<" expected "<<b[j]<<'\n';check(false);}}
int main(int argc,char**argv){try{
    check(argc==2||argc==3);
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()}){std::array<float,1> a{bad},b{0};rejects([&]{compare(a,b,"nonfinite actual");});rejects([&]{compare(b,a,"nonfinite expected");});}
    auto budget=std::make_shared<kadan::checkpoint::MemoryBudget>(16*1024*1024);
    if(argc==3){rejects([&]{kadan::checkpoint::ModelManifest manifest(argv[1],budget);auto g=kadan::model::read_generation(argv[1],manifest.architecture().vocab,budget);kadan::model::Layout l(manifest,8,g,budget);ModelImage image(l);kadan::model::load(l,std::make_shared<kadan::checkpoint::MemoryBudget>(64),image);});check(budget->used()==0);return 0;}
    kadan::checkpoint::ModelManifest manifest(argv[1],budget);auto generation=kadan::model::read_generation(argv[1],16,budget);check(generation.is_eos(15)&&generation.is_eos(14)&&!generation.is_eos(0));
    rejects([&]{kadan::model::Layout bad(manifest,9,generation,budget);});auto bad=generation;bad.eos[1]=15;rejects([&]{kadan::model::Layout l(manifest,8,bad,budget);});
    kadan::model::Layout layout(manifest,8,generation,budget);check(layout.layers().size()==manifest.architecture().layers);
    ModelImage image(layout),second(layout);const auto chunk=std::max(std::size_t(64),layout.minimum_staging_bytes());auto staging=std::make_shared<kadan::checkpoint::MemoryBudget>(chunk);kadan::model::load(layout,staging,image);check(staging->used()==0&&image.max_write<=chunk);
    kadan::model::load(layout,std::make_shared<kadan::checkpoint::MemoryBudget>(4096),second);check(image.arena==second.arena&&image.scales==second.scales&&image.writes>second.writes);
    rejects([&]{kadan::model::load(layout,std::make_shared<kadan::checkpoint::MemoryBudget>(8),second);});
    std::atomic_bool cancelled=true;auto before=second.writes;rejects([&]{kadan::model::load(layout,staging,second,&cancelled);});check(second.writes==before&&staging->used()==0);
    struct CancelSink:ModelImage {std::atomic_bool& flag;CancelSink(const kadan::model::Layout&l,std::atomic_bool&f):ModelImage(l),flag(f){}void write(std::size_t at,std::span<const std::uint8_t>d)override{ModelImage::write(at,d);flag=true;}};
    cancelled=false;CancelSink sink(layout,cancelled);rejects([&]{kadan::model::load(layout,staging,sink,&cancelled);});check(sink.writes==1&&staging->used()==0);
    auto embedding=image.dense("model.language_model.embed_tokens.weight"),norm=image.dense("model.language_model.norm.weight");auto head=image.matrix("lm_head");std::vector<std::unique_ptr<ModelCpuLayer>> layers;
    for(std::size_t i=0;i<layout.layers().size();++i)layers.push_back(std::make_unique<ModelCpuLayer>(image,i));
    const auto hidden=manifest.architecture().hidden;
    for(int replay=0;replay<2;++replay){std::ifstream expected(std::string(argv[1])+"/expected.bin",std::ios::binary);check(bool(expected));
        for(unsigned token:{2,7,11}){std::vector<float>x(hidden),y(hidden),n(hidden),logits(16),want(16);std::copy_n(embedding.begin()+hidden*token,hidden,x.begin());unsigned chosen=0;read(expected,std::span(&chosen,1));read(expected,std::span(want));
            for(std::size_t i=0;i<layers.size();++i){auto&ref=*layers[i]->reference;ref.step(x,y);x=y;std::vector<float> out(hidden);read(expected,std::span(out));compare(x,out,"layer");const auto&p=layout.layers()[i].plan;
                std::vector<std::uint8_t>a(p.state_first_bytes),b(p.state_second_bytes),ea(a.size()),eb(b.size());ref.read_state(a,b);read(expected,std::span(ea));read(expected,std::span(eb));check(a==ea);
                if(layout.layers()[i].config.attention==kadan::decoder::Attention::full)check(b==eb);else{std::vector<float> actual(b.size()/4),wanted(b.size()/4);std::memcpy(actual.data(),b.data(),b.size());std::memcpy(wanted.data(),eb.data(),eb.size());compare(actual,wanted,"recurrent",2e-5f);}}
            float sum=0;for(float v:x)sum+=v*v;float inv=1/std::sqrt(sum/hidden+1e-6f);for(std::size_t j=0;j<hidden;++j)n[j]=kadan::linear::bf16_round((x[j]*inv)*(1+norm[j]));
            for(std::size_t r=0;r<16;++r){double v=0;for(std::size_t j=0;j<hidden;++j)v+=double(kadan::model::bf16_weight(head,r,j))*n[j];logits[r]=kadan::linear::bf16_round(float(v));}
            compare(logits,want,"logits");check(unsigned(std::max_element(logits.begin(),logits.end())-logits.begin())==chosen);
        }check(expected.peek()==std::char_traits<char>::eof());for(auto& l:layers)l->reference->reset();}
    // Halfway rounding, signs and FP32-to-BF16 overflow. Synthetic powers of two
    // alone would fail to distinguish the checkpoint weight contract.
    std::array<std::uint8_t,16> w{};w.fill(0x38);float scale=1.00390625f;kadan::quantization::Matrix mat{kadan::quantization::Encoding::modelopt_fp8,1,16,w,{},std::span(&scale,1)};
    check(kadan::model::bf16_weight(mat,0,0)==1);scale=1.01171875f;check(kadan::model::bf16_weight(mat,0,0)==1.015625f);w[0]=0xb8;check(kadan::model::bf16_weight(mat,0,0)==-1.015625f);scale=std::numeric_limits<float>::max();rejects([&]{kadan::model::bf16_weight(mat,0,0);});
    // Atomic admission extension preserves the original ticket after failure.
    kadan::Resources r({100,100});auto a=r.reserve(kadan::Workload::llm,{60,0}),b=r.reserve(kadan::Workload::video,{20,30});rejects([&]{r.resize_loading(a,{81,71});});check(r.snapshot().used==kadan::Footprint({80,30}));r.resize_loading(a,{70,70});check(r.snapshot().used==kadan::Footprint({90,100}));r.loaded(a);rejects([&]{r.resize_loading(a,{0,0});});r.begin_eviction(a);r.released(a);r.released(b);check(r.snapshot().residents==0);
    // Mutation after a first successful chunk must invalidate the pinned shard.
    struct ChangedSink:ModelImage {std::string path;bool changed=false;char last=0;
        ChangedSink(const kadan::model::Layout&l,std::string p):ModelImage(l),path(std::move(p)){}
        void write(std::size_t at,std::span<const std::uint8_t>d)override{ModelImage::write(at,d);if(!changed){std::fstream f(path,std::ios::in|std::ios::out|std::ios::binary);f.seekg(-1,std::ios::end);f.read(&last,1);f.seekp(-1,std::ios::end);char v=last^1;f.write(&v,1);f.flush();check(bool(f));changed=true;}}
        ~ChangedSink(){if(changed){std::fstream f(path,std::ios::in|std::ios::out|std::ios::binary);f.seekp(-1,std::ios::end);f.write(&last,1);}}
    } changed(layout,std::string(argv[1])+"/fixture.safetensors");
    rejects([&]{kadan::model::load(layout,staging,changed);});check(changed.writes==1&&staging->used()==0);
    std::cout<<"Checkpoint CPU fixture: "<<layers.size()<<" layers; exact logits/layer outputs, recurrent tolerance 2e-5, bounded load and reset replay passed; arena="<<layout.device_bytes()<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
