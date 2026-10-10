#include "kadan/whisper_tokens.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("token_test_failed");}
template<class F>void rejects(F f,const char* error){try{f();}catch(const std::exception& e){check(e.what()==std::string(error));return;}throw std::runtime_error("expected_failure");}
int main(int argc,char** argv){
    check(argc==4);const std::string root=argv[1];const auto vocabulary=std::stoul(argv[2]),count=std::stoul(argv[3]);
    auto r=std::make_shared<Resources>(Footprint{32*1024*1024});stt::WhisperTokens tokens(r);std::atomic_bool cancel{false};
    rejects([&]{tokens.decode({});},"whisper_vocabulary_not_loaded");cancel=true;
    rejects([&]{tokens.load((root+"/english.tokens").c_str(),vocabulary,cancel);},"whisper_cancelled");cancel=false;
    tokens.load((root+"/english.tokens").c_str(),vocabulary,cancel);const auto used=r->snapshot().used;
    for(std::size_t i=0;i<count;++i){
        std::ifstream ids_file(root+"/"+std::to_string(i)+".ids",std::ios::binary|std::ios::ate);check(bool(ids_file));const auto size=ids_file.tellg();check(size>=0&&size<=448*4&&size%4==0);
        std::vector<std::uint32_t> ids(std::size_t(size)/4);ids_file.seekg(0);ids_file.read(reinterpret_cast<char*>(ids.data()),size);
        std::ifstream expected_file(root+"/"+std::to_string(i)+".expected",std::ios::binary);std::string expected((std::istreambuf_iterator<char>(expected_file)),{});
        if(tokens.decode(ids)!=expected)throw std::runtime_error("token_case_"+std::to_string(i));check(r->snapshot().used==used);
    }
    std::uint32_t bad=std::uint32_t(vocabulary);rejects([&]{tokens.decode({&bad,1});},"whisper_token_range");
    std::vector<std::uint32_t> large(449);rejects([&]{tokens.decode(large);},"whisper_text_bound");
    auto pressure=r->reserve(Workload::speech,{r->snapshot().capacity[0]-used[0]});std::uint32_t valid=0;
    rejects([&]{tokens.decode({&valid,1});},"exhausted");r->released(pressure);
    tokens.unload();tokens.unload();check(r->snapshot().residents==0);
    rejects([&]{tokens.load((root+"/truncated.tokens").c_str(),vocabulary,cancel);},"whisper_vocabulary_truncated");check(r->snapshot().residents==0);
    std::cout<<"Whisper token cases passed: "<<count<<'\n';
}
