#include "kadan/worker_compute.hpp"
#include <iostream>
#include <limits>
using namespace kadan;
void check(bool b){if(!b)throw std::runtime_error("compute_plan_test");}
template<class F>void fails(F f){try{f();}catch(const std::invalid_argument&){return;}throw std::runtime_error("expected_rejection");}
struct FakeContext final:DenseCompute {
 bool fail=false,scratch_fail=false;std::size_t releases=0,scratch_releases=0;
 std::shared_ptr<Resources> resources;Handle scratch=0;
 void release_scratch()override{++scratch_releases;if(scratch_fail)throw std::runtime_error("scratch_cleanup_unconfirmed");if(scratch){resources->released(scratch);scratch=0;}}
 void dense(std::span<const float>,std::span<const float>,std::span<const float>,std::size_t,std::size_t,std::span<float>,const std::atomic_bool&,bool)override{}
 void release_devices()override{++releases;if(fail)throw std::runtime_error("cleanup_unconfirmed");}
};
int main(){
 check(projection_split_columns(1,2));check(!projection_split_columns(2,2));check(!projection_split_columns(1,1));
 check(attention_visible(0,0,2,0,0));check(!attention_visible(0,1,2,0,0));
 check(attention_visible(2,3,2,0,0)); // Image rows see all keys after causal text.
 check(attention_visible(0,4,1,0,4));check(!attention_visible(0,5,1,0,4)); // Cached last query.
 check(!attention_visible(4,1,5,3,0));check(attention_visible(4,2,5,3,0));check(!attention_visible(4,5,5,3,0));

 auto cpu=compute_plan(1024,"","",false);check(cpu.capacity==Footprint{1024}&&cpu.devices.empty());
 const std::string budget=std::to_string(2ULL*1024*1024*1024+compute_context_bytes);
 auto dual=compute_plan(1024,"0,1",budget,true);check(dual.devices==std::vector<int>({0,1})&&dual.capacity==Footprint({1024,std::stoull(budget),std::stoull(budget)}));
 auto reverse=compute_plan(1024,"1,0",budget,true);check(reverse.devices==std::vector<int>({1,0}));
 auto second=compute_plan(1024,"1",budget,true);check(second.capacity[1]==0&&second.capacity[2]==std::stoull(budget));
 for(auto devices:{"0,0","2","-1","0,1,2","cpu","0,"})fails([&]{compute_plan(1024,devices,budget,true);});
 for(auto bytes:{"","0","268435456","-1","99999999999999999999999999","30000000000","536870912x"})fails([&]{compute_plan(1024,"0,1",bytes,true);});
 fails([&]{compute_plan(1024,"0,1",budget,false);});fails([&]{compute_plan(0,"","",true);});
 WorkerCompute owned(cpu,Workload::image);auto h=owned.resources->reserve(Workload::image,host_footprint(*owned.resources,1024));fails([&]{compute_plan(1024,"1","",true);});owned.resources->released(h);owned.idle();check(owned.resources->snapshot().used==Footprint{0});
 WorkerCompute parked(ComputePlan{{1024,1024},{}},Workload::tts);auto context=std::make_shared<FakeContext>();parked.compute=context;parked.context={0,512};parked.resume();check(parked.resources->snapshot().used==Footprint({0,512}));
 context->resources=parked.resources;context->scratch=parked.resources->reserve(Workload::tts,{0,128});
 context->scratch_fail=true;bool scratch_failed=false;try{parked.park();}catch(const std::runtime_error&){scratch_failed=true;}check(scratch_failed&&context->releases==0&&parked.context_handle&&parked.resources->snapshot().used[1]==640);context->scratch_fail=false;parked.idle();check(!context->scratch&&parked.resources->snapshot().used[1]==512);
 context->fail=true;bool failed=false;try{parked.park();}catch(const std::runtime_error&){failed=true;}check(failed&&parked.context_handle&&parked.resources->snapshot().used[1]==512);
 context->fail=false;parked.park();check(!parked.context_handle&&parked.resources->snapshot().used[1]==0);parked.resume();
 auto active=parked.resources->reserve(Workload::tts,{0,128});const auto calls=context->releases;failed=false;try{parked.park();}catch(const std::runtime_error&){failed=true;}check(failed&&context->releases==calls&&parked.resources->snapshot().used[1]==640);parked.resources->released(active);parked.park();check(parked.resources->snapshot().residents==0);
 std::cout<<"CPU device-selection, budget and fail-closed contracts passed\n";
}
