#pragma once
#include "kadan/image_generation.hpp"

namespace kadan::image {
// C++ operator backend owned by the existing serial image Generator.
class AtenGenerator {
public:
    AtenGenerator(std::shared_ptr<Resources>, std::vector<int>);
    ~AtenGenerator();
    void load(const std::string&, const std::atomic_bool&, const Generator::Hook&);
    void prepare(const ImageRequest&, const std::atomic_bool&);
    void generate(
        const ImageRequest&,
        std::span<float>,
        const std::atomic_bool&,
        const Generator::Hook&
    );
    void unload();

private:
    struct Impl;
    std::unique_ptr<Impl> model_;
    std::shared_ptr<Resources> resources_;
    std::vector<int> devices_;
};
}  // namespace kadan::image
