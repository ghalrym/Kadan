#include "kadan/checkpoint_floats.hpp"
#include "kadan/float_weights.hpp"
#include <array>
#include <iostream>
using namespace kadan::checkpoint;
void check(bool value){if(!value)throw std::runtime_error("test_failed");}
template<class F>void fails(F f){try{f();}catch(const std::exception&){return;}throw std::runtime_error("expected_failure");}
int main(){
 // Ownership/admission only, with no projection or model execution.
 kadan::Resources resources({64});
 {
  FloatSlice owned(resources,kadan::Workload::image,8);
  check(resources.snapshot().used[0]==32);
  FloatSlice moved(std::move(owned));check(moved.size()==8);
  fails([&]{FloatSlice excess(resources,kadan::Workload::image,9);});
  fails([&]{FloatSlice overflow(resources,kadan::Workload::image,std::numeric_limits<std::size_t>::max());});
  check(resources.snapshot().used[0]==32);
 }
 check(resources.snapshot().used[0]==0&&resources.snapshot().residents==0);
 std::atomic_bool cancel=false;
 const std::array<std::uint8_t,8> bytes{0,0,128,63,0,0,0,128};
 std::array<float,2> out{};std::array<std::uint8_t,5> scratch{};std::size_t reads=0,progress=0;
 auto read=[&](std::size_t offset,std::span<std::uint8_t> target){++reads;std::copy_n(bytes.begin()+offset,target.size(),target.begin());};
 read_floats(Dtype::fp32,out,scratch,cancel,read,[&](std::size_t n){progress=n;});
 check(reads==2&&progress==8&&out[0]==1&&std::signbit(out[1]));
 std::array<std::uint8_t,4> bf{128,63,0,128};reads=0;
 read_floats(Dtype::bf16,out,scratch,cancel,[&](std::size_t offset,std::span<std::uint8_t> target){++reads;std::copy_n(bf.begin()+offset,target.size(),target.begin());},[](auto){});
 check(reads==1&&out[0]==1&&std::signbit(out[1]));
 fails([&]{read_floats(Dtype::i8,out,scratch,cancel,read,[](auto){});});
 fails([&]{read_floats(Dtype::fp32,out,std::span(scratch).first(3),cancel,read,[](auto){});});
 cancel=true;reads=0;fails([&]{read_floats(Dtype::fp32,out,scratch,cancel,read,[](auto){});});check(!reads);
 cancel=false;fails([&]{read_floats(Dtype::fp32,out,scratch,cancel,[&](auto offset,auto target){read(offset,target);cancel=true;},[](auto){});});check(reads==1);
 cancel=false;fails([&]{read_floats(Dtype::fp32,out,scratch,cancel,[](auto,std::span<std::uint8_t> target){std::fill(target.begin(),target.end(),255);},[](auto){});});
 std::cout<<"bounded storage parsing and cancellation passed\n";
}
