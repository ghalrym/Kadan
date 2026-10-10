#include "kadan/image_tokenizer.hpp"
#include <array>
#include <iostream>
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("usage: kadan-image-tokenize TOKENIZER_JSON TEXT");
    auto r=std::make_shared<kadan::Resources>(kadan::Footprint{1024ULL*1024*1024});std::atomic_bool cancel=false;kadan::image::PromptTokenizer tokenizer(r);tokenizer.load(argv[1],cancel);
    const auto admission=r->reserve(kadan::Workload::image,{kadan::image::PromptTokenizer::max_tokens*sizeof(std::uint32_t)});
    {
        std::array<std::uint32_t,kadan::image::PromptTokenizer::max_tokens> ids;const auto count=tokenizer.encode(argv[2],ids,cancel);std::cout<<'[';for(std::size_t i=0;i<count;++i)std::cout<<(i?",":"")<<ids[i];std::cout<<"]\n";
    }
    r->released(admission);tokenizer.unload();if(r->snapshot().used[0])throw std::runtime_error("cleanup");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
