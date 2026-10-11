#include "kadan/whisper_tokens.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
namespace kadan::stt {
namespace {
void check(bool value,const char* text){if(!value)throw std::runtime_error(text);}
std::uint32_t number(std::ifstream& in){unsigned char b[4];in.read(reinterpret_cast<char*>(b),4);check(bool(in),"whisper_vocabulary_truncated");return std::uint32_t(b[0])|(std::uint32_t(b[1])<<8)|(std::uint32_t(b[2])<<16)|(std::uint32_t(b[3])<<24);}
std::string utf8(const std::string& bytes){
    std::string out;out.reserve(bytes.size()*3);
    for(std::size_t at=0;at<bytes.size();){const auto c=static_cast<unsigned char>(bytes[at]);std::size_t n=c<128?1:c>=194&&c<=223?2:c>=224&&c<=239?3:c>=240&&c<=244?4:0;
        bool valid=n&&at+n<=bytes.size();
        if(valid)for(std::size_t i=1;i<n;++i){auto d=static_cast<unsigned char>(bytes[at+i]);if(d<128||d>191)valid=false;}
        if(valid&&n>=3){auto d=static_cast<unsigned char>(bytes[at+1]);if((c==224&&d<160)||(c==237&&d>=160)||(c==240&&d<144)||(c==244&&d>=144))valid=false;}
        if(valid){out.append(bytes,at,n);at+=n;}else{
            std::size_t consumed=1;
            if(n)while(consumed<n&&at+consumed<bytes.size()){
                auto next=static_cast<unsigned char>(bytes[at+consumed]);
                if(next<128||next>191)break;
                if(consumed==1&&((c==224&&next<160)||(c==237&&next>=160)||(c==240&&next<144)||(c==244&&next>=144)))break;
                ++consumed;
            }
            out+="\xef\xbf\xbd";at+=consumed;
        }
    }
    return out;
}
}
WhisperTokens::WhisperTokens(std::shared_ptr<Resources> r):resources_(std::move(r)){check(bool(resources_),"whisper_resources_required");}
WhisperTokens::~WhisperTokens(){unload();}
void WhisperTokens::load(const char* path,std::size_t vocabulary,const std::atomic_bool& cancel){
    check(!resident_,"whisper_vocabulary_loaded");check(!cancel.load(),"whisper_cancelled");check(vocabulary>0&&vocabulary<=52000,"whisper_vocabulary_size");
    auto f=resources_->snapshot().capacity;std::fill(f.begin(),f.end(),0);f[0]=16*1024*1024;resident_=resources_->reserve(Workload::speech,std::move(f));
    try{
        std::ifstream in(path,std::ios::binary|std::ios::ate);check(bool(in)&&in.tellg()>0&&in.tellg()<=8*1024*1024,"whisper_vocabulary_file");in.seekg(0);
        char magic[8];in.read(magic,8);check(std::string_view(magic,8)=="KDWVOC01","whisper_vocabulary_format");check(number(in)==vocabulary,"whisper_vocabulary_size");eos_=number(in);check(eos_<vocabulary,"whisper_token_range");
        auto list=[&](std::vector<std::uint32_t>& ids,std::size_t cap){const auto count=number(in);check(count<=cap,"whisper_vocabulary_bound");ids.resize(count);for(auto& id:ids){id=number(in);check(id<vocabulary,"whisper_token_range");}};
        list(prompt_,4);list(suppressed_,vocabulary);list(first_,8);check(!prompt_.empty(),"whisper_vocabulary_prompt");
        pieces_.reserve(vocabulary);for(std::size_t i=0;i<vocabulary;++i){check(!cancel.load(),"whisper_cancelled");auto n=number(in);check(n<=128,"whisper_vocabulary_bound");std::string piece(n,'\0');in.read(piece.data(),n);check(bool(in),"whisper_vocabulary_truncated");pieces_.push_back(std::move(piece));}
        check(in.peek()==std::char_traits<char>::eof(),"whisper_vocabulary_trailing");
    }catch(...){unload();throw;}
}
void WhisperTokens::unload(){if(!resident_)return;std::vector<std::string>().swap(pieces_);std::vector<std::uint32_t>().swap(prompt_);std::vector<std::uint32_t>().swap(suppressed_);std::vector<std::uint32_t>().swap(first_);resources_->released(resident_);resident_=0;eos_=0;}
std::string WhisperTokens::decode(std::span<const std::uint32_t> ids)const{
    check(resident_!=0,"whisper_vocabulary_not_loaded");check(ids.size()<=448,"whisper_text_bound");
    auto footprint=resources_->snapshot().capacity;std::fill(footprint.begin(),footprint.end(),0);footprint[0]=ids.size()*128;
    struct Scratch{Resources& r;Handle h;~Scratch(){r.released(h);}} scratch{*resources_,resources_->reserve(Workload::speech,std::move(footprint))};
    std::string bytes;bytes.reserve(ids.size()*128);
    for(auto id:ids){check(id<pieces_.size(),"whisper_token_range");if(id==eos_)break;check(id<eos_,"whisper_generated_special_token");bytes+=pieces_[id];}
    return utf8(bytes);
}
}
