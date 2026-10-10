#include "kadan/h3_compute.hpp"
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <vector>
using namespace kadan;
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("expected one authorized device");
    const int device=std::stoi(argv[1]);
    auto resources=std::make_shared<Resources>(Footprint{64ULL<<20,512ULL<<20,512ULL<<20});
    auto host=resources->reserve(Workload::video,{8ULL<<20,0,0});
    std::atomic_bool cancel=false;
    auto compute=video::h3_cuda_compute(resources,{device});
    // Unequal work across warps exposes maximum-to-sum shared-memory reuse.
    // Repetition alone is not a race proof: also run under CUDA racecheck.
    for(bool precise:{false,true})for(std::size_t keys:{31,33,137,257,4101}){
        const std::size_t queries=keys,heads=2,kv=1,dim=8;
        std::vector<float> q(queries*heads*dim),k(keys*kv*dim),v(k.size()),y(q.size());
        for(std::size_t i=0;i<q.size();++i)q[i]=std::sin(float(i)*.19f)*4;
        for(std::size_t i=0;i<k.size();++i){k[i]=std::cos(float(i)*.07f);v[i]=std::sin(float(i)*.05f);}
        for(unsigned repeat=0;repeat<8;++repeat){
            compute->attention(q,k,v,queries,heads,kv,dim,false,y,cancel,precise);
            for(std::size_t t:std::array<std::size_t,3>{0,queries/2,queries-1})for(std::size_t h=0;h<heads;++h){
                std::vector<double> p(keys);double maximum=-INFINITY,total=0;
                for(std::size_t j=0;j<keys;++j){double value=0;for(std::size_t c=0;c<dim;++c)value+=double(q[(t*heads+h)*dim+c])*k[j*dim+c];p[j]=value/std::sqrt(double(dim));maximum=std::max(maximum,p[j]);}
                for(auto& value:p){value=std::exp(value-maximum);total+=value;}
                for(std::size_t c=0;c<dim;++c){double expected=0;for(std::size_t j=0;j<keys;++j)expected+=p[j]/total*v[j*dim+c];if(!std::isfinite(y[(t*heads+h)*dim+c])||std::abs(y[(t*heads+h)*dim+c]-expected)>3e-5+3e-5*std::abs(expected))throw std::runtime_error("attention mismatch");}
            }
        }
    }
    compute.reset();resources->released(host);
    if(resources->snapshot().used!=Footprint({0,0,0}))throw std::runtime_error("reservation leak");
    std::cout<<"PASS 80 actual CUDA attention calls, F32/F64, uneven warps and tile crossing; reservations zero\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
