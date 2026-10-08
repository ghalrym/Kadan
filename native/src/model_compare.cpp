#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace {
void require(bool x,const char* e){if(!x)throw std::runtime_error(e);}
float tolerance(const char* s){float v=0;std::string_view t(s);auto[e,c]=std::from_chars(t.data(),t.data()+t.size(),v);require(c==std::errc{}&&e==t.data()+t.size()&&std::isfinite(v)&&v>=0,"invalid_tolerance");return v;}
struct Capture {
    std::ifstream file;std::array<std::uint32_t,4> header{},record{};std::vector<float> logits;
    explicit Capture(const char* path):file(path,std::ios::binary){read(header.data(),sizeof(header));require(header[0]==0x4b4d4331&&header[1]==1&&header[2]>0&&header[2]<=262144&&header[3]>0&&header[3]<=8,"capture_header");logits.resize(header[2]);}
    void read(void* p,std::size_t n){file.read(static_cast<char*>(p),std::streamsize(n));require(bool(file),"capture_truncated");}
    void next(std::size_t i){read(record.data(),sizeof(record));require(record[0]<header[2]&&record[1]<header[2]&&record[2]<=1&&record[3]==i+1,"capture_record");read(logits.data(),logits.size()*sizeof(float));for(float v:logits)require(std::isfinite(v),"capture_nonfinite");require(std::size_t(std::max_element(logits.begin(),logits.end())-logits.begin())==record[1],"capture_greedy_selection");}
    void finish(){std::uint32_t done=0;read(&done,4);require(done==0x444f4e45&&file.peek()==std::char_traits<char>::eof(),"capture_incomplete_or_trailing");}
};
}
int main(int argc,char**argv){if(argc!=5){std::cerr<<"Usage: kadan-model-compare EXPECTED ACTUAL ABS_TOL REL_TOL\n";return 2;}try{
    static_assert(std::endian::native==std::endian::little);float absolute=tolerance(argv[3]),relative=tolerance(argv[4]);Capture expected(argv[1]),actual(argv[2]);require(expected.header==actual.header,"capture_shape_mismatch");std::size_t mismatches=0;double maximum=0;
    for(std::size_t i=0;i<expected.header[3];++i){expected.next(i);actual.next(i);require(expected.record==actual.record,"capture_token_or_eos_mismatch");for(std::size_t j=0;j<expected.logits.size();++j){double e=expected.logits[j],a=actual.logits[j],error=std::abs(e-a);maximum=std::max(maximum,error);if(error>double(absolute)+double(relative)*std::abs(e))++mismatches;}}
    expected.finish();actual.finish();std::cout<<"max_abs_error="<<maximum<<" outside_tolerance="<<mismatches<<" abs_tol="<<absolute<<" rel_tol="<<relative<<'\n';require(mismatches==0,"logit_parity_failed");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
