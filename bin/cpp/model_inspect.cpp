#include "kadan/model_manifest.hpp"
#include <array>
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
std::size_t number(const char* text) {
    std::size_t n=0; const std::string_view value(text);
    const auto [end,error]=std::from_chars(value.data(),value.data()+value.size(),n);
    if (error!=std::errc{} || end!=value.data()+value.size()) throw std::invalid_argument("invalid_budget_argument");
    return n;
}
}
int main(int argc,char** argv) {
    if (argc!=3 && argc!=9) {
        std::cerr<<"Usage: kadan-model-inspect ROOT METADATA_BYTES [HOST_BYTES STAGING_BYTES DEVICE_BYTES HEADROOM_BYTES EXPERT_SLOTS SPLIT_LAYER]\n";
        return 2;
    }
    try {
        auto budget=std::make_shared<kadan::checkpoint::MemoryBudget>(number(argv[2]));
        kadan::checkpoint::ModelManifest model(argv[1],budget);
        const auto& a=model.architecture();
        std::cout<<"layers="<<a.layers<<" experts="<<a.experts<<" hidden="<<a.hidden<<" vocab="<<a.vocab
                 <<" tensors="<<model.tensor_count()<<" excluded="<<model.excluded_tensor_count()
                 <<" items="<<model.items().size()<<" checkpoint_bytes="<<model.checkpoint_bytes()
                 <<" metadata_used="<<budget->used()<<'\n';
        if (argc==9) {
            const std::array<std::size_t,2> capacities{number(argv[5]),number(argv[5])};
            const std::array<std::size_t,2> headroom{number(argv[6]),number(argv[6])},slots{number(argv[7]),number(argv[7])};
            const auto split=number(argv[8]);
            if (!split || split>=a.layers) throw std::invalid_argument("split_layer");
            std::vector<std::size_t> layers(a.layers); for (std::size_t i=0;i<a.layers;++i) layers[i]=i<split?0:1;
            const auto plan=model.place({capacities,headroom,slots,layers,0,number(argv[3]),number(argv[4])});
            std::cout<<"host_bytes="<<plan.host_bytes<<" expert_host_bytes="<<plan.expert_host_bytes
                     <<" metadata_envelope="<<plan.metadata_bytes<<" staging_bytes="<<plan.staging_bytes
                     <<" max_transfer_bytes="<<plan.max_transfer_bytes<<'\n';
            for (std::size_t i=0;i<plan.devices().size();++i) {
                const auto& d=plan.devices()[i];
                std::cout<<"device="<<i<<" resident_bytes="<<d.resident_bytes<<" expert_cache_bytes="<<d.expert_cache_bytes
                         <<" headroom_bytes="<<d.headroom_bytes<<" total_bytes="<<d.total_bytes<<'\n';
            }
        }
        std::cout<<"metadata_only: no payloads read, no allocation of model weights, no CUDA access\n";
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
