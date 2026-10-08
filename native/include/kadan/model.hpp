#pragma once
#include "kadan/decoder.hpp"
#include <atomic>
namespace kadan::model {
// Text-only, batch-one, weight-only BF16 contract. No tokenizer or sampling.
struct Generation {
    std::array<unsigned,16> eos{};std::size_t count=0;
    bool is_eos(unsigned id)const;
};
Generation read_generation(const char* root,std::size_t vocabulary,const std::shared_ptr<checkpoint::MemoryBudget>&);
struct Layer {decoder::Config config;decoder::Plan plan;std::size_t offset;};
struct Binding {
    std::size_t item,weights,scales;
    int layer,expert,part; // NVFP4 scalar destination: layer=-1 means head.
};
// Immutable validated bindings. Borrows the manifest, which must outlive it.
class Layout {
public:
    Layout(const checkpoint::ModelManifest&,std::size_t capacity,Generation,
           std::shared_ptr<checkpoint::MemoryBudget>);
    Layout(const Layout&)=delete;Layout&operator=(const Layout&)=delete;
    const checkpoint::ModelManifest& manifest()const{return manifest_;}
    std::span<const Layer> layers()const{return layers_;}
    std::span<const Binding> bindings()const{return bindings_;}
    std::size_t capacity()const{return capacity_;}
    std::size_t minimum_staging_bytes()const{return minimum_staging_;}
    const Generation& generation()const{return generation_;}
    std::size_t device_bytes()const{return end_;}
    std::size_t embedding()const{return embedding_;}std::size_t final_norm()const{return final_norm_;}
    std::size_t head_weights()const{return head_weights_;}std::size_t head_scales()const{return head_scales_;}
    std::size_t embedded()const{return embedded_;}std::size_t normalized()const{return normalized_;}
    std::size_t logits()const{return logits_;}std::size_t selected()const{return selected_;}std::size_t status()const{return status_;}
private:
    const checkpoint::ModelManifest& manifest_;
    std::shared_ptr<checkpoint::MemoryBudget> budget_;
    std::pmr::vector<Layer> layers_;std::pmr::vector<Binding> bindings_;
    std::size_t capacity_,minimum_staging_=2,end_=0,embedding_,final_norm_,head_weights_,head_scales_,embedded_,normalized_,logits_,selected_,status_;
    Generation generation_;
};
class Sink {
public:
    virtual ~Sink()=default;
    // Blocking consumption; borrowed bytes expire when this call returns.
    virtual void write(std::size_t offset,std::span<const std::uint8_t>)=0;
    virtual void multiplier(const Binding&,float)=0;
};
// Reads finite BF16 dense data and canonical packed projection rows in a bounded
// staging envelope. Sink must remain unpublished on ANY failure. Does not allocate
// an arena, create a reservation, decode a full matrix or execute CUDA.
void load(const Layout&,std::shared_ptr<checkpoint::MemoryBudget> staging,Sink&,
          const std::atomic_bool* cancelled=nullptr);
// Same FP32 decode then BF16 round-to-nearest-even as the existing weight-only
// Python reference. Used for admission and CPU fixture/reference checks.
float bf16_weight(const quantization::Matrix&,std::size_t row,std::size_t column);
} // namespace kadan::model
