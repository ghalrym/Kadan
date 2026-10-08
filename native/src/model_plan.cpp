#include "kadan/model.hpp"
#include <charconv>
#include <iostream>
#include <string_view>
std::size_t number(const char* s){std::size_t n=0;std::string_view v(s);auto[e,c]=std::from_chars(v.data(),v.data()+v.size(),n);if(c!=std::errc{}||e!=v.data()+v.size())throw std::invalid_argument("invalid_number");return n;}
int main(int argc,char**argv){if(argc!=4){std::cerr<<"Usage: kadan-model-plan ROOT CAPACITY METADATA_BYTES (metadata only)\n";return 2;}try{
    auto budget=std::make_shared<kadan::checkpoint::MemoryBudget>(number(argv[3]));kadan::checkpoint::ModelManifest m(argv[1],budget);auto g=kadan::model::read_generation(argv[1],m.architecture().vocab,budget);kadan::model::Layout l(m,number(argv[2]),g,budget);std::cout<<"layers="<<l.layers().size()<<" bound_items="<<l.bindings().size()<<" vocabulary="<<m.architecture().vocab<<" arena_bytes="<<l.device_bytes()<<" minimum_staging_bytes="<<l.minimum_staging_bytes()<<" metadata_used="<<budget->used()<<" eos=";for(std::size_t i=0;i<g.count;++i)std::cout<<(i?",":"")<<g.eos[i];std::cout<<"\nmetadata_only: no tensor payloads, CUDA calls or model allocation\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
