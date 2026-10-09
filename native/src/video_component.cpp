#include "kadan/video.hpp"
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: kadan-video-component VAE_ROOT SHARD INPUT_F32LE OUTPUT_COMPONENT");
        // Bounded CPU diagnostic; never probes CUDA or starts inference runners.
        std::ifstream input(argv[3], std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("input_open");
        auto bytes=input.tellg();
        if (bytes<=0 || bytes>24*4*kadan::video::H3DecoderInput::max_tokens || bytes%(24*4)) throw std::runtime_error("input_shape_or_limit");
        auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{8*1024*1024});
        struct InputReservation {
            std::shared_ptr<kadan::Resources> ledger; kadan::Handle handle;
            ~InputReservation() { ledger->released(handle); }
        };
        {
        InputReservation admission{resources, resources->reserve(kadan::Workload::video,{static_cast<kadan::Bytes>(bytes)})};
        std::vector<float> values(static_cast<std::size_t>(bytes)/4);
        input.seekg(0); input.read(reinterpret_cast<char*>(values.data()),bytes);
        if (!input) throw std::runtime_error("input_read");
        std::atomic_bool cancel{false};
        kadan::video::H3DecoderInput stage(resources);
        stage.load(argv[1],argv[2],cancel);
        stage.execute(values,argv[4],cancel);
        stage.unload();
        std::vector<float>().swap(values);
        } // Input allocation is destroyed before its reservation on every exit.
        std::cout << "H3 decoder-input component executed; full_video_generation=false; gpu_execution=false; resident_bytes=" << resources->snapshot().used[0] << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
