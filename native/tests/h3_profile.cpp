#include "kadan/h3_profile.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace kadan::video;
void check(bool b){if(!b)throw std::runtime_error("h3_profile_test_failed");}
int main(){try{
    // Independent expected values from the installed H3 shape resolver, for
    // every advertised duration and each API resolution/aspect combination.
    constexpr std::array<std::size_t,12> frames{107,124,158,175,192,226,243,277,294,328,345,362};
    constexpr std::array<std::size_t,12> times{32,37,47,52,57,67,72,82,87,97,102,107};
    for(auto edge:{480u,768u})for(auto aspect:{"16:9","9:16","1:1"})for(std::size_t seconds=4;seconds<=15;++seconds){
        const auto p=h3_api_profile(edge,aspect,seconds);const auto long_edge=edge==480?864u:1344u;
        check(p.width==(std::string_view(aspect)=="16:9"?long_edge:edge));check(p.height==(std::string_view(aspect)=="9:16"?long_edge:edge));
        check(p.frames==frames[seconds-4]&&p.video_time==times[seconds-4]);check(p.frames>=seconds*24&&p.frames-seconds*24<17);check(p.video_tokens==p.video_time*(p.width/32)*(p.height/32));
    }
    const auto maximum=h3_api_profile(768,"16:9",15);check(maximum.video_tokens==107856&&maximum.audio_tokens==1206&&maximum.vae_tokens==28224);
    for(auto edge:{0u,128u,720u,1080u}){bool rejected=false;try{h3_api_profile(edge,"1:1",4);}catch(const std::invalid_argument&){rejected=true;}check(rejected);}
    for(auto seconds:{0u,3u,16u,~0u}){bool rejected=false;try{h3_api_profile(768,"1:1",seconds);}catch(const std::invalid_argument&){rejected=true;}check(rejected);}
    std::cout<<"PASS all 72 advertised H3 profiles, temporal alignment and maximum token counts\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
