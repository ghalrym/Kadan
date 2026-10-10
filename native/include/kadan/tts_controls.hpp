#pragma once
#include "kadan/tts_generator.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
namespace kadan::tts {
// Checkpoint-owned control IDs, independent of numerical execution. Small bounded
// metadata is covered by the worker's process reservation before construction.
class VoiceControls {
    std::map<std::string,std::uint32_t> languages_,speakers_;
    std::map<std::string,std::string> dialects_;
    VoicePrompt prompt_;
    static void need(bool ok){if(!ok)throw std::runtime_error("tts_voice_config");}
    static std::uint32_t id(const nlohmann::json& value){need(value.is_number_unsigned());auto n=value.get<std::uint64_t>();need(n<3072);return std::uint32_t(n);}
    static std::string lower(std::string value){for(char& c:value){need(static_cast<unsigned char>(c)<128);c=char(std::tolower(static_cast<unsigned char>(c)));}return value;}
public:
    explicit VoiceControls(const std::string& path){
        std::ifstream f(path,std::ios::binary|std::ios::ate);need(bool(f)&&f.tellg()>0&&f.tellg()<=65536);
        std::string data(std::size_t(f.tellg()),'\0');f.seekg(0);f.read(data.data(),data.size());need(bool(f));
        const auto root=nlohmann::json::parse(data),j=root.at("talker_config");
        prompt_.think=id(j.at("codec_think_id"));prompt_.think_bos=id(j.at("codec_think_bos_id"));prompt_.think_eos=id(j.at("codec_think_eos_id"));prompt_.nothink=id(j.at("codec_nothink_id"));
        prompt_.codec_pad=id(j.at("codec_pad_id"));prompt_.codec_bos=id(j.at("codec_bos_id"));prompt_.codec_eos=id(j.at("codec_eos_token_id"));
        need(j.at("codec_language_id").is_object()&&j.at("codec_language_id").size()<=32&&j.at("spk_id").is_object()&&j.at("spk_id").size()<=32);
        for(const auto& [key,value]:j.at("codec_language_id").items())languages_.emplace(lower(key),id(value));
        for(const auto& [key,value]:j.at("spk_id").items())speakers_.emplace(lower(key),id(value));
        for(const auto& [key,value]:j.at("spk_is_dialect").items()){
            need(speakers_.contains(key));if(value.is_boolean()){need(value==false);continue;}
            need(value.is_string()&&languages_.contains(value.get<std::string>()));dialects_.emplace(key,value.get<std::string>());
        }
    }
    VoicePrompt resolve(std::string speaker,std::string language)const{
        speaker=lower(speaker);language=lower(language);need(speakers_.contains(speaker));
        need(language=="auto"||languages_.contains(language));
        if((language=="auto"||language=="chinese")&&dialects_.contains(speaker))language=dialects_.at(speaker);
        auto result=prompt_;result.speaker=speakers_.at(speaker);result.automatic_language=language=="auto";
        if(!result.automatic_language)result.language=languages_.at(language);
        return result;
    }
};
}
