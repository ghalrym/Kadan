#include "kadan/h3_generation.hpp"
#include "kadan/h3_audio.hpp"
#include "h3_codec.hpp"
#include "kadan/h3_tokenizer.hpp"
#include "kadan/h3_text.hpp"
#include "kadan/h3_denoiser.hpp"
#include "kadan/video.hpp"
#include "kadan/weight_progress.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <unistd.h>
#include <fcntl.h>
namespace kadan::video {
namespace {
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void stop(const std::atomic_bool& cancel){check(!cancel.load(),"h3_generation_cancelled");}
struct Active { bool& b; explicit Active(bool& value):b(value){check(!b,"busy");b=true;} ~Active(){b=false;} };
struct Admission { std::shared_ptr<Resources> r; Handle id; Admission(std::shared_ptr<Resources> ledger,Bytes n):r(std::move(ledger)){auto f=r->snapshot().capacity;std::fill(f.begin(),f.end(),0);f[0]=n;id=r->reserve(Workload::video,std::move(f));} ~Admission(){r->released(id);} };
struct Workspace {
    std::string dir;
    explicit Workspace(const std::string& output){dir=(std::filesystem::path(output).parent_path()/".kadan-h3-XXXXXX").string();check(mkdtemp(dir.data())!=nullptr,"h3_generation_temp");}
    ~Workspace(){for(const auto* name:{"features","frames","video","encoded","audio"})unlink((dir+"/"+name).c_str());rmdir(dir.c_str());}
    std::string path(const char* name)const{return dir+"/"+name;}
};
void read_tensor(const std::string& path,const std::string& header,std::span<float> out){
    std::ifstream in(path,std::ios::binary);std::string actual(header.size(),'\0');in.read(actual.data(),actual.size());check(actual==header,"h3_generation_tensor_header");in.read(reinterpret_cast<char*>(out.data()),out.size_bytes());check(bool(in)&&in.peek()==std::char_traits<char>::eof(),"h3_generation_tensor_bytes");for(float v:out)check(std::isfinite(v),"h3_generation_nonfinite");
}
template<class Model> void load(Model& model,const std::string& path,const std::atomic_bool& cancel,const std::shared_ptr<checkpoint::ReadCache>& cache){const std::filesystem::path p(path);model.load(p.parent_path().c_str(),p.filename().string(),cancel,cache);}
void write_audio(const std::string& path,std::span<const float> samples,const std::atomic_bool& cancel){
    check(samples.size()>0&&samples.size()<=2*640*800,"h3_audio_output_shape");
    std::ofstream out(path,std::ios::binary|std::ios::trunc);check(bool(out),"h3_audio_output_open");
    auto word=[&](std::uint32_t value,std::size_t n){for(std::size_t i=0;i<n;++i)out.put(char(value>>(8*i)));};
    out.write("RIFF",4);word(36+samples.size()*2,4);out.write("WAVEfmt ",8);word(16,4);word(1,2);word(2,2);word(32000,4);word(128000,4);word(4,2);word(16,2);out.write("data",4);word(samples.size()*2,4);
    for(std::size_t i=0;i<samples.size();++i){if(i%4096==0)stop(cancel);check(std::isfinite(samples[i]),"h3_audio_output_nonfinite");word(std::uint16_t(std::clamp(std::lround(samples[i]*32768),-32768L,32767L)),2);}
    out.close();check(bool(out),"h3_audio_output_write");
}
void noise(std::span<float> output,std::mt19937_64& random){
    // Explicit Box-Muller transform, independent of STL normal_distribution.
    constexpr double tau=6.283185307179586476925286766559;
    for(std::size_t i=0;i<output.size();i+=2){const double a=(double(random()>>11)+0.5)/9007199254740992.0,b=(double(random()>>11)+0.5)/9007199254740992.0;const double radius=std::sqrt(-2*std::log(a));output[i]=float(radius*std::cos(tau*b));if(i+1<output.size())output[i+1]=float(radius*std::sin(tau*b));}
}
}
namespace h3 {
void temporal_join(std::span<float> raw,std::span<float> tail,std::size_t plane,bool previous){
    check(plane>0&&plane<=1344*768&&raw.size()==3*28*plane&&tail.size()==3*5*plane,"h3_generation_temporal_shape");
    for(std::size_t c=0;c<3;++c)for(std::size_t f=0;f<5;++f)for(std::size_t p=0;p<plane;++p){
        const auto dst=(c*28+3+f)*plane+p,cache=(c*5+f)*plane+p;
        if(previous){const float weight=float(f)/5;raw[dst]=tail[cache]*(1-weight)+raw[dst]*weight;}
        tail[cache]=raw[(c*28+23+f)*plane+p];
    }
}

std::vector<float> sigmas(std::size_t updates,float shift){
    check(updates>0&&updates<=64&&std::isfinite(shift)&&shift>0,"h3_generation_schedule");std::vector<float> out(updates+1);
    // Match the F32 symmetric linspace endpoints, then F32 time shifting.
    const float step=1.0f/float(updates);
    for(std::size_t i=0;i<=updates;++i){const float base=i<=(updates/2)?1.0f-float(i)*step:float(updates-i)*step;out[i]=(shift*base)/(1.0f+(shift-1.0f)*base);}
    return out;
}
void timesteps(std::span<float> output,std::size_t text_rows,std::size_t video_rows,float video_sigma,float audio_sigma){
    check(text_rows>0&&video_rows>0&&text_rows<=output.size()&&video_rows<=output.size()-text_rows,"h3_generation_timestep_shape");
    check(std::isfinite(video_sigma)&&std::isfinite(audio_sigma)&&video_sigma>=0&&video_sigma<=1&&audio_sigma>=0&&audio_sigma<=1,"h3_generation_timestep_sigma");
    std::fill(output.begin(),output.begin()+text_rows+video_rows,1-video_sigma);
    std::fill(output.begin()+text_rows+video_rows,output.end(),1-audio_sigma);
}
void advance(std::span<float> state,std::span<const float> velocity,float current,float next){
    check(state.size()==velocity.size()&&std::isfinite(current)&&std::isfinite(next)&&current>0&&current<=1&&next>=0&&next<=current,"h3_generation_step");const float ratio=next/current;const float time=1.0f-current;
    for(std::size_t i=0;i<state.size();++i){const float denoised=state[i]+(1.0f-time)*velocity[i];const float value=ratio*state[i]+(1.0f-ratio)*denoised;check(std::isfinite(value),"h3_generation_nonfinite");state[i]=value;}
}
void unpack(std::span<const float> in,std::span<float> out,std::size_t t,std::size_t h,std::size_t w){
    check(t>0&&t<=107&&h>0&&h<=84&&w>0&&w<=84&&h*w<=4032&&h%2==0&&w%2==0&&in.size()==t*h*w*24&&out.size()==in.size(),"h3_generation_unpack");
    for(std::size_t z=0;z<t;++z)for(std::size_t y=0;y<h;++y)for(std::size_t x=0;x<w;++x)for(std::size_t c=0;c<24;++c)out[((z*h+y)*w+x)*24+c]=in[((z*(h/2)+y/2)*(w/2)+x/2)*96+c*4+(y%2)*2+x%2];
}
}
std::shared_ptr<checkpoint::ReadCache> h3_weight_cache(std::shared_ptr<Resources> resources,
    const H3GenerationPaths& paths,Bytes limit,const std::atomic_bool& cancel,std::shared_ptr<H3Compute> compute,
    const std::function<void(const char*,std::size_t)>& hook){
    stop(cancel);if(!limit&&!compute)return {};
    checkpoint::WeightInventory inventory;
    {
        Admission metadata(resources,32*1024*1024);
        auto budget=std::make_shared<checkpoint::MemoryBudget>(32*1024*1024);
        for(const auto* path:{&paths.text,&paths.denoiser,&paths.turbo,&paths.vae,&paths.audio_vae}){
            stop(cancel);if(path->empty())continue;const std::filesystem::path file(*path);
            checkpoint::Limits limits;limits.metadata_value_bytes=8192;
            checkpoint::Shard shard(file.parent_path().c_str(),file.filename().string(),budget,limits);
            inventory.add(shard);
        }
    }
    stop(cancel);
    if(compute){
        const auto execution_bytes=checkpoint::weight_bytes_add(inventory.floating_f32_bytes,inventory.raw_nonfloating_bytes);
        compute->prepare_weights(execution_bytes);
        report_weight_inventory(inventory,hook);report_weight_placement(*compute,hook);
    }
    if(!limit)return {};
    // Raw integer and decoded float weights share one envelope, including metadata.
    const auto selected=std::min(limit,checkpoint::weight_bytes_add(std::max(inventory.stored_bytes,checkpoint::weight_bytes_add(inventory.floating_f32_bytes,inventory.raw_nonfloating_bytes)),8*1024*1024));
    return std::make_shared<checkpoint::ReadCache>(std::move(resources),selected,Workload::video);
}
H3Generation::H3Generation(std::shared_ptr<Resources> resources,std::shared_ptr<H3Compute> compute,std::shared_ptr<checkpoint::ReadCache> cache):resources_(std::move(resources)),compute_(std::move(compute)),cache_(std::move(cache)){check(bool(resources_),"h3_generation_resources");}
void H3Generation::execute(const H3GenerationPaths& paths,const H3GenerationRequest& request,const std::atomic_bool& cancel,const Hook& hook){
    static_assert(std::endian::native==std::endian::little);stop(cancel);Active active(busy_);
    const std::size_t axis_limit=compute_?1344:128;
    check(request.width>=32&&request.width<=axis_limit&&request.width%32==0&&request.height>=32&&request.height<=axis_limit&&request.height%32==0&&request.width*request.height<=1344*768,"h3_generation_dimensions");
    check(request.frames>=22&&request.frames<=(compute_?362:90)&&(request.frames-5)%17==0,"h3_generation_frames");check(!request.output.empty()&&request.output.size()<=4096&&request.output.find('\0')==std::string::npos,"h3_generation_output");
    check(!std::filesystem::exists(request.output)&&!std::filesystem::is_symlink(request.output),"h3_generation_output_exists");
    const bool mp4=std::filesystem::path(request.output).extension()==".mp4";
    check(mp4||std::filesystem::path(request.output).extension()==".y4m","h3_generation_output_format");
    const auto video_sigmas=h3::sigmas(request.updates,12),audio_sigmas=h3::sigmas(request.updates,3);
    const auto t=(request.frames-5)/17*5+2,h=request.height/16,w=request.width/16,nv=t*h*w/4;
    // 24 fps and 40 Hz audio latents. Integer ties-to-even matches round().
    const auto audio_num=request.frames*5,audio_q=audio_num/3,at=audio_q+(audio_num%3>=2);const auto na=2*at;
    check(nv+na+1<=(compute_?107856+1206+H3Tokenizer::max_tokens:H3Denoiser::max_tokens),"h3_generation_token_limit");
    // Bound all caller tensors/control buffers before allocation. Scratch and
    // streamed weights are separately admitted by each executor. Only one
    // feature artifact (<=10MiB) or decoder chunk (<=348MiB) exists at a time.
    const Bytes caller_bytes=32ULL*1024*1024+(3*nv*96+2*na*32+7*h*w*24)*4+request.width*request.height*(3*33*4+3)+(512+nv+na)*20;
    Admission caller(resources_,caller_bytes);Workspace workspace(request.output);
    std::array<std::uint32_t,H3Tokenizer::max_tokens> ids{};H3Tokenizer tokenizer(resources_);tokenizer.load(paths.tokenizer,cancel);const auto nt=tokenizer.encode(request.prompt,ids,cancel);tokenizer.unload();check(nt+nv+na<=(compute_?107856+1206+H3Tokenizer::max_tokens:H3Denoiser::max_tokens),"h3_generation_token_limit");
    auto actual_caller=resources_->snapshot().capacity;std::fill(actual_caller.begin(),actual_caller.end(),0);actual_caller[0]=caller_bytes+nt*5120*4;resources_->resize_loading(caller.id,std::move(actual_caller));
    if(hook)hook("tokenized",nt);
    std::vector<float> text(nt*5120);H3TextEncoder text_model(resources_,compute_);load(text_model,paths.text,cancel,cache_);text_model.execute({ids.data(),nt},workspace.path("features"),cancel,hook);text_model.unload();
    read_tensor(workspace.path("features"),"KADAN_H3_CONDITIONING_F32_V1\n"+std::to_string(nt)+" 5120\nF32LE\n",text);check(unlink(workspace.path("features").c_str())==0,"h3_generation_temp_cleanup");
    std::vector<float> video(nv*96),audio(na*32),dv(video.size()),da(audio.size());std::mt19937_64 random(request.seed);noise(video,random);noise(audio,random);
    const auto n=nt+nv+na;std::vector<float> positions(n*3),times(n,1);std::vector<std::uint32_t> tags(n,1);
    for(std::size_t i=0;i<nt;++i)positions[i*3]=float(i);
    const double area=std::sqrt(double(h*w));auto axis=[&](std::size_t index,std::size_t dim){const double ratio=double(dim)/area;return float(((1-ratio)/2+double(index)*ratio/double(dim/2))*32);};
    double time=double(nt);for(std::size_t z=0;z<t;++z){for(std::size_t y=0;y<h/2;++y)for(std::size_t x=0;x<w/2;++x){auto i=nt+(z*(h/2)+y)*(w/2)+x;positions[i*3]=float(time);positions[i*3+1]=axis(y,h);positions[i*3+2]=axis(x,w);tags[i]=0;}time+=(z%5==0?1:4)*(5.0/3.0);}
    for(std::size_t channel=0;channel<2;++channel)for(std::size_t z=0;z<at;++z){auto i=nt+nv+channel*at+z;positions[i*3]=float(nt+z);positions[i*3+2]=axis(channel? w/2-1:0,w);tags[i]=2;}
    H3Denoiser denoiser(resources_,compute_);load(denoiser,paths.denoiser,cancel,cache_);const std::filesystem::path adapter(paths.turbo);denoiser.load_turbo(adapter.parent_path().c_str(),adapter.filename().string(),cancel,cache_);
    for(std::size_t step=0;step<request.updates;++step){stop(cancel);h3::timesteps(times,nt,nv,video_sigmas[step],audio_sigmas[step]);denoiser.execute({text,video,audio,positions,times,tags},dv,da,cancel,hook);h3::advance(video,dv,video_sigmas[step],video_sigmas[step+1]);h3::advance(audio,da,audio_sigmas[step],audio_sigmas[step+1]);if(hook)hook("denoise_completed",step+1);}
    denoiser.unload();
    if(!paths.audio_vae.empty()){
        Admission audio_output(resources_,at*800*2*sizeof(float)+8192);
        std::vector<float> waveform(at*800*2);H3AudioDecoder audio_decoder(resources_);
        const std::filesystem::path asset(paths.audio_vae);audio_decoder.load(asset.parent_path().c_str(),asset.filename().string(),{},cancel,cache_);
        audio_decoder.decode(audio,at,waveform,cancel,[&](const char* phase,std::size_t index){if(hook)hook((std::string("audio_")+phase).c_str(),index);});audio_decoder.unload();write_audio(workspace.path("audio"),waveform,cancel);if(hook)hook("audio_decoded",at*800);
    }
    std::vector<float> latent(video.size());h3::unpack(video,latent,t,h,w);
    H3VideoDecoder decoder(resources_,compute_);load(decoder,paths.vae,cancel,cache_);std::vector<float> chunk(7*h*w*24),frames(3*28*request.height*request.width),tail(3*5*request.height*request.width);std::vector<unsigned char> pixels(3*request.height*request.width);
    std::ofstream output(workspace.path("video"),std::ios::binary|std::ios::trunc);check(bool(output),"h3_generation_output_open");output<<"YUV4MPEG2 W"<<request.width<<" H"<<request.height<<" F24:1 Ip A1:1 C444 XCOLORRANGE=FULL\n";
    std::size_t written=0;const auto plane=request.width*request.height;constexpr float means[]={.485f,.456f,.406f},stds[]={.229f,.224f,.225f};
    auto write_frame=[&](std::size_t f){stop(cancel);for(std::size_t p=0;p<plane;++p){float rgb[3];for(std::size_t c=0;c<3;++c)rgb[c]=std::clamp(frames[(c*28+f)*plane+p]*stds[c]+means[c],0.0f,1.0f)*255;auto byte=[](float v){return static_cast<unsigned char>(std::clamp(std::round(v),0.0f,255.0f));};pixels[p]=byte(.299f*rgb[0]+.587f*rgb[1]+.114f*rgb[2]);pixels[plane+p]=byte(128-.168736f*rgb[0]-.331264f*rgb[1]+.5f*rgb[2]);pixels[2*plane+p]=byte(128+.5f*rgb[0]-.418688f*rgb[1]-.081312f*rgb[2]);}output<<"FRAME\n";output.write(reinterpret_cast<const char*>(pixels.data()),pixels.size());check(bool(output),"h3_generation_output_write");++written;};
    // Released token_drop=3 profile: T=5*n+2, n>=1. Each seven-token
    // window advances five tokens; aligned requests need no repeated padding.
    for(std::size_t start=0;start+7<=t;start+=5){stop(cancel);std::copy_n(latent.data()+start*h*w*24,chunk.size(),chunk.data());
        decoder.execute(chunk,7,h,w,workspace.path("frames"),cancel,hook);read_tensor(workspace.path("frames"),"KADAN_H3_FRAMES_V1\n3 28 "+std::to_string(request.height)+" "+std::to_string(request.width)+"\nF32LE\n",frames);check(unlink(workspace.path("frames").c_str())==0,"h3_generation_temp_cleanup");
        h3::temporal_join(frames,tail,plane,start!=0);
        for(std::size_t f=3;f<20;++f)write_frame(f);
        if(start+7==t)for(std::size_t f=23;f<28;++f)write_frame(f);
    }
    decoder.unload();check(written==request.frames,"h3_generation_frame_count");output.close();check(bool(output),"h3_generation_output_close");stop(cancel);
    std::string final=workspace.path("video");if(mp4){Admission codec(resources_,256ULL*1024*1024);h3::encode_mp4(final,workspace.path("encoded"),cancel,request.frames,paths.audio_vae.empty()?std::string{}:workspace.path("audio"));final=workspace.path("encoded");}
    int fd=open(final.c_str(),O_RDONLY|O_CLOEXEC);check(fd>=0,"h3_generation_output_sync");const int sync=fsync(fd);close(fd);check(sync==0,"h3_generation_output_sync");stop(cancel);
    if(compute_)report_weight_placement(*compute_,hook);
    const auto sidecar=request.output+".wav";bool audio_published=false;
    if(!mp4&&!paths.audio_vae.empty()){check(link(workspace.path("audio").c_str(),sidecar.c_str())==0,"h3_audio_output_publish");audio_published=true;}
    if(link(final.c_str(),request.output.c_str())!=0){if(audio_published)unlink(sidecar.c_str());throw std::runtime_error("h3_generation_output_publish");}
}
}
