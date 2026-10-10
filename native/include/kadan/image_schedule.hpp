#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace kadan::image {
// Checkpoint scheduler: exponential dynamic shift, terminal stretch to .02,
// deterministic Euler integration. One update is invalid for terminal stretch.
struct Schedule { std::array<float,101> sigma{}; std::size_t steps=0; };
inline Schedule schedule(std::size_t tokens,std::size_t steps){
 if(tokens==0||tokens>16384||steps<2||steps>100)throw std::invalid_argument("image_schedule_bounds");
 Schedule s;s.steps=steps;const double mu=.5+(double(tokens)-256)*(.9-.5)/(8192-256);const float e=float(std::exp(mu));
 for(std::size_t i=0;i<steps;++i){float t=float(1.+(1./double(steps)-1.)*double(i)/double(steps-1));s.sigma[i]=e/(e+(1.f/t-1.f));}
 const float scale=(1.f-s.sigma[steps-1])/(1.f-.02f);
 for(std::size_t i=0;i<steps;++i)s.sigma[i]=1.f-(1.f-s.sigma[i])/scale;
 s.sigma[steps]=0;return s;
}
}
