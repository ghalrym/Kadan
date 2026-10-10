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

 check(projection_columns(3072,96768,4,true,1)<96768);
 check(projection_columns(3072,96768,1,false,4)%4==0);
 check(projection_columns(24,24,4,false,1)==24);
 fails([]{projection_columns(2147483647,4,4,true,1);});
 // Pure workspace arithmetic: independently include every aligned CUDA slice.
 const auto aligned=[](std::initializer_list<Bytes> sizes){Bytes n=0;for(auto size:sizes){n=(n+255)/256*256;n+=size;}return (n+255)/256*256;};
 for(Bytes limit:{16ULL*1024*1024,64ULL*1024*1024,2ULL*1024*1024*1024}){
  for(std::size_t width:{1U,4U})for(bool precise:{false,true})for(std::size_t in:{24U,3072U,4096U,16384U}){
   const auto plan=projection_plan(in,96768,width,precise,width==1?4:1,limit);
   const Bytes t=plan.rows,o=plan.columns,w=in*o;
   const auto bytes=aligned({w*width,o*4,o*4,t*in*4,t*in*4,t*in,t*4,t*o*4,t*o*4,8*1024*1024,precise?w*8:0,precise?t*in*8:0,precise?t*o*8:0});
   check(bytes<=plan.bytes&&plan.bytes<=limit&&o>0&&t>0&&t<=256&&(width!=1||o%4==0));
  }
  for(std::size_t width:{4U,8U})for(std::size_t n:{1U,129U,18432U})for(std::size_t keys:{3U,4097U,18432U}){
   const auto plan=attention_plan(n,keys,2048,width,limit);const Bytes q=plan.queries,k=plan.keys;
   const Bytes qs=(plan.full_heads?n:q)*2048,ks=(plan.full_heads?keys:k)*2048;
   check(aligned({qs*width,ks*width,ks*width,q*2048*width,q*k*width,q*width,q*width,8*1024*1024})<=plan.bytes&&plan.bytes<=limit);
   std::size_t queries_seen=0,keys_seen=0;
   for(std::size_t start=0;start<n;start+=q)queries_seen+=std::min<std::size_t>(q,n-start);
   for(std::size_t start=0;start<keys;start+=k)keys_seen+=std::min<std::size_t>(k,keys-start);
   check(queries_seen==n&&keys_seen==keys);
  }
 }
 const auto normal=projection_plan(4096,4096,4,false,1,2ULL*1024*1024*1024);
 check(normal.rows==256&&normal.columns==4096);
 const auto small=projection_plan(4096,4096,4,false,1,16ULL*1024*1024);check(small.bytes<=16ULL*1024*1024&&small.columns<4096);
 check(attention_plan(18432,18432,1024,4,2ULL*1024*1024*1024).full_heads);
 check(!attention_plan(18432,18432,1024,4,16ULL*1024*1024).full_heads);
 fails([]{attention_plan(1,1,1,4,8*1024*1024);});
 fails([]{projection_plan(4096,4096,4,false,1,8*1024*1024);});
 unsigned attempts=0,pauses=0;bool cancellation=false;
 wait_for_device_allocation([&]{return cancellation;},[&]{return ++attempts==3;},[&]{++pauses;});check(attempts==3&&pauses==2);
 attempts=0;pauses=0;bool interrupted=false;
 try{wait_for_device_allocation([&]{return cancellation;},[&]{++attempts;return false;},[&]{++pauses;cancellation=true;});}catch(const std::runtime_error&){interrupted=true;}
 check(interrupted&&attempts==1&&pauses==1);cancellation=false;
 interrupted=false;try{wait_for_device_allocation([]{return false;},[]{throw std::runtime_error("device_fault");return false;},[&]{++pauses;});}catch(const std::runtime_error&){interrupted=true;}check(interrupted&&pauses==1);
 auto cpu=compute_plan(1024,"","",false);check(cpu.capacity==Footprint{1024}&&cpu.devices.empty());
 const std::string budget=std::to_string(2ULL*1024*1024*1024+compute_context_bytes);
 auto dual=compute_plan(1024,"0,1",budget,true);check(dual.devices==std::vector<int>({0,1})&&dual.capacity==Footprint({1024,std::stoull(budget),std::stoull(budget)}));
 const auto large=std::to_string(8ULL*1024*1024*1024);
 auto unequal=compute_plan(1024,"1,0",budget,true,"0:"+budget+",1:"+large);check(unequal.capacity[1]==std::stoull(budget)&&unequal.capacity[2]==std::stoull(large));
 for(auto mapping:{"0:4294967296","0:4294967296,0:4294967296","0:4294967296,1:4294967296,","2:4294967296,0:4294967296","0:1,1:4294967296"})fails([&]{compute_plan(1024,"0,1",budget,true,mapping);});
 fails([&]{compute_plan(1024,"0",budget,true,"0:4294967296,1:4294967296");});
 const std::array<Bytes,2> capacities{1,3},reverse_capacities{3,1},zero{0,0};
 check(weight_boundary(100,capacities,1)==25&&weight_boundary(100,reverse_capacities,1)==75);
 check(weight_boundary(100,zero,1)==50&&weight_boundary(100,capacities,2)==100);
 for(std::size_t n=0;n<100;++n){auto mid=weight_boundary(n,capacities,1);check(mid<=n&&mid*4+(n-mid)*4==n*4);}
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
