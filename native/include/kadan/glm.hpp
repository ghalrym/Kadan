#pragma once
#include "kadan/model_worker.hpp"
#include "kadan/worker_compute.hpp"
#include <map>
namespace kadan::glm {
// Text-only glm5_next/NVFP4. Vision and MTP are deliberately not dispatched.
inline constexpr bool is_eos(unsigned token){return token==154820||token==154827||token==154829;}
struct Plan {std::size_t context,host,packed,tensors,execution_weights;};
Plan plan(const char*,std::size_t);
std::size_t state_bytes(std::size_t);
class Engine final:public serving::ResidentEngine {
public:
 Engine(const char*,Plan,const ComputePlan&);
 ~Engine();
 serving::Info info()const override;
 void begin_request()override;void end_request()override;void park()override;
 void reset()override;serving::Token step(unsigned,bool)override;void close()override;
 std::shared_ptr<Resources> resources()const;
private:struct Impl;std::unique_ptr<Impl> impl_;
};
}
