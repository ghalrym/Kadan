// Synthetic checkpoint headers only: no weights decoded, model load or
// execution.
#include "kadan/h3_generation.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
void check(bool value) {
  if (!value)
    throw std::runtime_error("test_failed");
}
int main() {
  char folder[] = "/tmp/kadan-h3-header-XXXXXX";
  check(mkdtemp(folder));
  struct Cleanup {
    const char *p;
    ~Cleanup() { std::filesystem::remove_all(p); }
  } cleanup{folder};
  auto write = [&](std::size_t bytes) {
    std::string header =
        "{\"__metadata__\":{\"config\":\"" + std::string(bytes, 'x') +
        "\"},\"w\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
    std::ofstream f(std::string(folder) + "/weights.safetensors",
                    std::ios::binary);
    for (unsigned i = 0; i < 8; ++i)
      f.put(char(std::uint64_t(header.size()) >> (i * 8)));
    f << header;
    f.put(0);
    f.put(0);
  };
  auto resources =
      std::make_shared<kadan::Resources>(kadan::Footprint{64 * 1024 * 1024});
  std::atomic_bool cancel = false;
  kadan::video::H3GenerationPaths paths;
  paths.text = std::string(folder) + "/weights.safetensors";
  for (auto length : {521, 1654, 2013, 8192}) {
    write(length);
    auto cache =
        kadan::video::h3_weight_cache(resources, paths, 1024 * 1024, cancel);
    check(bool(cache));
    cache.reset();
    check(resources->snapshot().used[0] == 0);
  }
  write(8193);
  bool rejected = false;
  try {
    auto cache =
        kadan::video::h3_weight_cache(resources, paths, 1024 * 1024, cancel);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected && resources->snapshot().used[0] == 0);
  std::cout << "Bounded H3 inventory metadata parsing passed (no inference)\n";
}
