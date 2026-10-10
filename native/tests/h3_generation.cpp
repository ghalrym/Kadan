#include "kadan/h3_generation.hpp"
#include "kadan/h3_tokenizer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>
using namespace kadan;
using namespace kadan::video;
void check(bool b){if(!b)throw std::runtime_error("test_failed");}
template<class F>void fails(F f,const char* expected){try{f();throw std::logic_error("expected_failure");}catch(const std::runtime_error& e){check(std::string(e.what())==expected);}}
int main(int argc,char** argv){try{
  const auto s=h3::sigmas(8,12); const std::vector<float> expected{1,.9882352948188782f,.9729729890823364f,.9523809552192688f,.9230769276618958f,.8780487775802612f,.800000011920929f,.6315789222717285f,0};check(s==expected);check(h3::sigmas(8,3)==std::vector<float>({1,.9545454382896423f,.8999999761581421f,.8333333134651184f,.75f,.6428571343421936f,.5f,.30000001192092896f,0}));check(s.size()==9&&s.front()==1&&s.back()==0);for(std::size_t i=1;i<s.size();++i)check(s[i]<s[i-1]);check(std::abs(s[4]-12.0f/13)<1e-7f);fails([]{h3::sigmas(0,12);},"h3_generation_schedule");fails([]{h3::sigmas(8,std::numeric_limits<float>::quiet_NaN());},"h3_generation_schedule");
  // Released packed-sequence contract: text and video share time, audio differs.
  std::array<float,7> times;times.fill(-7);h3::timesteps(times,2,3,.75f,.25f);
  check(times==std::array<float,7>{.25f,.25f,.25f,.25f,.25f,.75f,.75f});
  h3::timesteps(times,2,3,1,1);check(std::all_of(times.begin(),times.end(),[](float v){return v==0;}));
  const auto unchanged=times;
  fails([&]{h3::timesteps(times,8,1,.5f,.5f);},"h3_generation_timestep_shape");check(times==unchanged);
  fails([&]{h3::timesteps(times,2,3,std::numeric_limits<float>::quiet_NaN(),.5f);},"h3_generation_timestep_sigma");check(times==unchanged);
  fails([&]{h3::timesteps(times,2,3,.5f,1.1f);},"h3_generation_timestep_sigma");check(times==unchanged);
  std::array<float,2> state{2,-3},velocity{4,6};h3::advance(state,velocity,1,.5f);check(state[0]==4&&state[1]==0);h3::advance(state,velocity,.5f,0);check(state[0]==6&&state[1]==3);fails([&]{h3::advance(state,velocity,0,.1f);},"h3_generation_step");
  std::vector<float> in(2*4*6*24),out(in.size());for(std::size_t z=0;z<2;++z)for(std::size_t py=0;py<2;++py)for(std::size_t px=0;px<3;++px)for(std::size_t c=0;c<24;++c)for(std::size_t dy=0;dy<2;++dy)for(std::size_t dx=0;dx<2;++dx)in[((z*2+py)*3+px)*96+c*4+dy*2+dx]=float(z*10000+(py*2+dy)*1000+(px*2+dx)*100+c);
  h3::unpack(in,out,2,4,6);for(std::size_t z=0;z<2;++z)for(std::size_t y=0;y<4;++y)for(std::size_t x=0;x<6;++x)for(std::size_t c=0;c<24;++c)check(out[((z*4+y)*6+x)*24+c]==float(z*10000+y*1000+x*100+c));
  // Independent planar oracle: two clips yield 17+17+5=39 frames.
  std::vector<float> raw(3*28*2),tail(3*5*2);
  for(std::size_t i=0;i<raw.size();++i)raw[i]=float(i);
  const auto original=raw;h3::temporal_join(raw,tail,2,false);check(raw==original);
  const auto previous=tail;std::fill(raw.begin(),raw.end(),1000);
  h3::temporal_join(raw,tail,2,true);
  for(std::size_t c=0;c<3;++c)for(std::size_t f=0;f<5;++f)for(std::size_t p=0;p<2;++p){const float w=float(f)/5;check(raw[(c*28+3+f)*2+p]==previous[(c*5+f)*2+p]*(1-w)+1000*w);check(tail[(c*5+f)*2+p]==1000);}
  fails([&]{h3::temporal_join(raw,tail,1,true);},"h3_generation_temporal_shape");
  char dir[]="/tmp/kadan-generation-test-XXXXXX";check(mkdtemp(dir)!=nullptr);const auto root=std::filesystem::path(dir);auto r=std::make_shared<Resources>(Footprint{1024ULL*1024*1024});H3Generation gen(r);H3GenerationRequest req;req.prompt="test";req.output=(root/"video.y4m").string();std::atomic_bool cancel=true;
  fails([&]{gen.execute({},req,cancel);},"h3_generation_cancelled");cancel=false;req.width=63;fails([&]{gen.execute({},req,cancel);},"h3_generation_dimensions");req.width=64;req.frames=5;fails([&]{gen.execute({},req,cancel);},"h3_generation_frames");req.frames=22;
  fails([&]{gen.execute({},req,cancel);},"h3_tokenizer_file_size");check(r->snapshot().used[0]==0&&std::filesystem::is_empty(root));
  auto small=std::make_shared<Resources>(Footprint{1024});H3Generation tight(small);fails([&]{tight.execute({},req,cancel);},"exhausted");check(small->snapshot().used[0]==0&&std::filesystem::is_empty(root));
  {std::ofstream occupied(req.output);occupied<<"preserve";} fails([&]{gen.execute({},req,cancel);},"h3_generation_output_exists");check(std::filesystem::file_size(req.output)==8);std::filesystem::remove(req.output);
  H3Tokenizer tok(r);auto bad=(root/"bad.json").string();{std::ofstream f(bad);f<<"{\"x\":1,\"x\":2}";}fails([&]{tok.load(bad,cancel);},"h3_tokenizer_duplicate_key");check(r->snapshot().used[0]==0);std::filesystem::remove(bad);
  if(argc==2){tok.load(argv[1],cancel);check(r->snapshot().used[0]==192ULL*1024*1024);std::array<std::uint32_t,512> ids{};check(tok.encode("A red ball.",ids,cancel)==4);check(ids[0]==32&&ids[1]==2518&&ids[2]==4935&&ids[3]==13);cancel=true;fails([&]{tok.encode("hello",ids,cancel);},"h3_tokenizer_cancelled");cancel=false;fails([&]{tok.encode(std::string("\xff",1),ids,cancel);},"h3_tokenizer_utf8");fails([&]{tok.encode("hello",std::span<std::uint32_t>{},cancel);},"h3_tokenizer_token_limit");check(tok.encode("hello",ids,cancel)>0);tok.unload();check(r->snapshot().used[0]==0);}
  std::filesystem::remove(root);std::cout<<"PASS schedule, layout, cancellation, failed admission, cleanup, tokenizer recovery\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
