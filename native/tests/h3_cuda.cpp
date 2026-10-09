#include "kadan/h3_compute.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
using namespace kadan;using namespace kadan::video;
void check(bool b){if(!b)throw std::runtime_error("h3_cuda_test_failed");}
template<class F>void fails(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed);}
void near(float a,double b){check(std::isfinite(a)&&std::abs(double(a)-b)<=3e-5+3e-5*std::abs(b));}
int main(int argc,char** argv){try{
    check(argc>=2);std::vector<int> devices;for(int i=1;i<argc;++i)devices.push_back(std::stoi(argv[i]));
    auto r=std::make_shared<Resources>(Footprint{1024ULL*1024*1024,2ULL*1024*1024*1024,2ULL*1024*1024*1024});
    // Test fixture host tensors are admitted before allocation; CUDA context
    // residency belongs to this short-lived test process, not the model ledger.
    auto host=r->reserve(Workload::video,{128ULL*1024*1024,0,0});std::atomic_bool cancel=false;
    fails([&]{h3_cuda_compute(r,{});});fails([&]{h3_cuda_compute(r,{0,0});});fails([&]{h3_cuda_compute(r,{2});});
    auto compute=h3_cuda_compute(r,devices);
    {std::array<float,1>x{1},w{1};fails([&]{compute->dense(w,{},x,1,1,x,cancel);});cancel=true;std::array<float,1>y{};fails([&]{compute->dense(w,{},x,1,1,y,cancel);});cancel=false;}
    for(std::size_t group:{64,256}){
        const std::size_t in=group*2,out=20,rows=257;
        std::vector<float> x(rows*in),scales(out),bias(out),y(rows*out);std::vector<std::uint8_t>w(in*out);
        for(std::size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.071f)*.5f;
        for(std::size_t i=0;i<w.size();++i)w[i]=std::bit_cast<std::uint8_t>(std::int8_t(int(i%255)-128));
        for(std::size_t i=0;i<out;++i){scales[i]=.001f+float(i)*.0001f;bias[i]=float(i)*.01f;}
        compute->convrot(w,scales,bias,x,in,out,group,y,cancel);
        std::vector<double> rotated(in);std::vector<int> codes(in);
        for(std::size_t t=0;t<rows;++t){for(std::size_t c=0;c<in;++c)rotated[c]=x[t*in+c];
            for(std::size_t g=0;g<in;g+=group)for(std::size_t stride=1;stride<group;stride*=4)for(std::size_t a=0;a<group;a+=stride*4)for(std::size_t j=0;j<stride;++j){auto* p=rotated.data()+g+a+j;const double v0=p[0],v1=p[stride],v2=p[2*stride],v3=p[3*stride];p[0]=((v0+v1)+(v2-v3))*.5;p[stride]=((v0+v1)+(v3-v2))*.5;p[2*stride]=((v0+v2)+(v3-v1))*.5;p[3*stride]=((v1+v2)+(v3-v0))*.5;}
            float maximum=1e-10f;for(double& v:rotated){v=float(v);maximum=std::max(maximum,std::abs(float(v)));}const float scale=maximum/127;
            for(std::size_t c=0;c<in;++c){const float z=float(rotated[c])/scale,lo=std::floor(z);int code=int(lo);if(z-lo>.5f||(z-lo==.5f&&code%2))++code;codes[c]=std::clamp(code,-127,127);}
            for(std::size_t o=0;o<out;++o){std::int32_t sum=0;for(std::size_t c=0;c<in;++c)sum+=std::bit_cast<std::int8_t>(w[o*in+c])*codes[c];check(y[t*out+o]==(float(sum)*scale)*scales[o]+bias[o]);}
        }
    }
    {const std::size_t rows=259,in=37,out=23;std::vector<float>w(in*out),x(rows*in),b(out),y(rows*out);for(std::size_t i=0;i<w.size();++i)w[i]=std::cos(float(i)*.13f)*.1f;for(std::size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.11f);for(std::size_t i=0;i<out;++i)b[i]=float(i)*.01f;compute->dense(w,b,x,in,out,y,cancel);for(std::size_t t=0;t<rows;++t)for(std::size_t o=0;o<out;++o){double v=b[o];for(std::size_t c=0;c<in;++c)v+=double(w[o*in+c])*x[t*in+c];near(y[t*out+o],v);}}
    for(bool precise:{false,true})for(bool causal:{false,true})for(std::size_t n:{137,4101}){
        const std::size_t heads=4,kv=2,dim=8;std::vector<float>q(n*heads*dim),k(n*kv*dim),v(k.size()),y(q.size());for(std::size_t i=0;i<q.size();++i)q[i]=std::sin(float(i)*.19f);for(std::size_t i=0;i<k.size();++i){k[i]=std::cos(float(i)*.07f);v[i]=std::sin(float(i)*.05f);}
        compute->attention(q,k,v,n,heads,kv,dim,causal,y,cancel,precise);
        for(std::size_t t:std::array<std::size_t,5>{0,1,127,n/2,n-1})for(std::size_t h=0;h<heads;++h){std::vector<double>p(n);double max=-INFINITY,total=0;const auto keys=causal?t+1:n;for(std::size_t j=0;j<keys;++j){double s=0;for(std::size_t c=0;c<dim;++c)s+=double(q[(t*heads+h)*dim+c])*k[(j*kv+h/2)*dim+c];p[j]=s/std::sqrt(double(dim));max=std::max(max,p[j]);}for(std::size_t j=0;j<keys;++j){p[j]=std::exp(p[j]-max);total+=p[j];}for(std::size_t c=0;c<dim;++c){double s=0;for(std::size_t j=0;j<keys;++j)s+=p[j]/total*v[(j*kv+h/2)*dim+c];near(y[(t*heads+h)*dim+c],s);}}
    }
    {std::vector<float>w(64*64,.1f),x(100000*64,.2f),y(x.size());std::atomic_bool done=false;
        std::thread cancellation([&]{while(!done.load()){const auto used=r->snapshot().used;if(used[1]||used[2]){cancel=true;return;}std::this_thread::yield();}});
        bool cancelled=false;try{compute->dense(w,{},x,64,64,y,cancel);}catch(const std::exception&){cancelled=true;}done=true;cancellation.join();check(cancelled);cancel=false;
    }
    {auto tight=std::make_shared<Resources>(Footprint{1024,1,1});auto constrained=h3_cuda_compute(tight,devices);std::array<float,1>a{1},b{};fails([&]{constrained->dense(a,{},a,1,1,b,cancel);});check(tight->snapshot().used==Footprint({0,0,0}));}
    compute.reset();r->released(host);check(r->snapshot().used==Footprint({0,0,0}));
    std::cout<<"PASS INT8 convrot exact, dense and tiled grouped attention parity, cancellation and failed admission; final reservations zero\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
