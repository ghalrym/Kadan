#pragma once
#include "kadan/dense_compute.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
namespace kadan::image {
struct ImageRequest { std::string prompt;std::size_t width=128,height=128,steps=4;std::uint64_t seed=0; };
// Complete checkpoint-specific text-to-image pipeline, single CPU image. Editing
// and CFG are not supported by this boundary. Executors own separate pipelines.
class Generator {
public:
 using Hook=std::function<void(const char*,std::size_t)>;
 explicit Generator(std::shared_ptr<Resources>,std::shared_ptr<DenseCompute> compute={});
 ~Generator();
 void load(const std::string& root,const std::atomic_bool&,const Hook& = {});
 void generate(const ImageRequest&,std::span<float> rgba_chw,const std::atomic_bool&,const Hook& = {});
 void unload();
 static void validate(const ImageRequest&);
private:
 struct Impl;std::shared_ptr<Resources> resources_;std::shared_ptr<DenseCompute> compute_;std::unique_ptr<Impl> model_;bool busy_=false;
};
// Writes an owned PNG with exclusive creation. Caller admits publication memory.
void publish_png(const std::string&,std::span<const float> rgba_chw,std::size_t height,std::size_t width,const std::atomic_bool&);
}
