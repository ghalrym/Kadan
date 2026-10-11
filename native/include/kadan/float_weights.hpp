#pragma once
#include "kadan/checkpoint.hpp"
#include "kadan/dense_compute.hpp"
#include "kadan/worker_compute.hpp"
#include <algorithm>
#include <cmath>
#include <utility>
#include <limits>
#include <climits>
#if defined(KADAN_IMAGE_BLAS) || defined(KADAN_WHISPER_BLAS)
#include <cblas.h>
#endif
namespace kadan::checkpoint {
// Owned slice: borrowing a span never outlives its admission or decoded bytes.
class FloatSlice {
 Resources* resources_;Handle handle_;std::vector<float> values_;
static Bytes bytes(std::size_t count){if(count>std::numeric_limits<Bytes>::max()/sizeof(float))throw std::overflow_error("float_slice_size");return count*sizeof(float);}
public:
 FloatSlice(Resources& r,Workload w,std::size_t count):resources_(&r),handle_(r.reserve(w,host_footprint(r,bytes(count)))){try{values_.resize(count);}catch(...){r.released(handle_);throw;}}
 ~FloatSlice(){std::vector<float>().swap(values_);if(resources_)resources_->released(handle_);}
 FloatSlice(const FloatSlice&)=delete;
 FloatSlice(FloatSlice&& other)noexcept:resources_(std::exchange(other.resources_,nullptr)),handle_(other.handle_),values_(std::move(other.values_)){}
 std::span<float> span(){return values_;}
 operator std::span<const float>()const{return values_;}
 const float* data()const{return values_.data();}
 std::size_t size()const{return values_.size();}
 float operator[](std::size_t i)const{return values_[i];}
 auto begin()const{return values_.begin();}auto end()const{return values_.end();}
};
inline FloatSlice float_slice(Resources& r,Workload w,const Shard& shard,std::string_view name,std::size_t first,std::size_t count,const std::atomic_bool& cancel){FloatSlice result(r,w,count);shard.read_float_tensor(name,first,result.span(),cancel);return result;}
// Both GPU and CPU paths consume output-row tiles. GPU owns its separately
// bounded source staging; no full dense weight matrix is created by this helper.
static inline void project_source(Resources& r,Workload workload,DenseCompute* compute,const Shard& shard,std::string_view name,
 std::span<const float> bias,std::span<const float> input,std::size_t in,std::size_t out,std::span<float> output,const std::atomic_bool& cancel){
 if(!in||!out||input.size()%in||output.size()%out||output.size()/out!=input.size()/in||(!bias.empty()&&bias.size()!=out))throw std::invalid_argument("source_projection_shape");
 auto source=[&](std::size_t first,std::span<float> dst){shard.read_float_tensor(name,first,dst,cancel);};
 shard.check_unchanged();
 if(compute&&compute->dense_source(shard.tensor_identity(name),source,bias,input,in,out,output,cancel)){shard.check_unchanged();return;}
 const auto rows=input.size()/in;
 for(std::size_t first=0;first<out;first+=64){
  auto count=std::min<std::size_t>(64,out-first);auto weights=float_slice(r,workload,shard,name,first*in,count*in,cancel);
  for(std::size_t t=0;t<rows;t+=64){
   if(cancel.load())throw std::runtime_error("source_projection_cancelled");
   auto nr=std::min<std::size_t>(64,rows-t);
#if defined(KADAN_IMAGE_BLAS) || defined(KADAN_WHISPER_BLAS)
   if(in>INT_MAX||out>INT_MAX)throw std::overflow_error("source_projection_blas_shape");
   cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,int(nr),int(count),int(in),1,input.data()+t*in,int(in),weights.data(),int(in),0,output.data()+t*out+first,int(out));
   if(!bias.empty())for(std::size_t row=0;row<nr;++row)for(std::size_t o=0;o<count;++o)output[(t+row)*out+first+o]+=bias[first+o];
#else
   for(std::size_t row=0;row<nr;++row)for(std::size_t o=0;o<count;++o){float value=bias.empty()?0:bias[first+o];for(std::size_t c=0;c<in;++c)value+=input[(t+row)*in+c]*weights[o*in+c];output[(t+row)*out+first+o]=value;}
#endif
  }
 }
 shard.check_unchanged();
}
}
