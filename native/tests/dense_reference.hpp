#pragma once
#include "kadan/dense_compute.hpp"
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>
// CPU dispatch oracle only. It does not emulate or claim CUDA execution.
struct RecordingDense final : kadan::DenseCompute {
 std::size_t calls=0,attention_calls=0;bool fail=false;
 bool attend(std::span<const float> q,std::span<const float> k,std::span<const float> v,std::size_t rows,std::size_t keys,std::size_t heads,std::size_t kv,std::size_t dim,std::size_t causal,std::size_t window,std::size_t offset,std::span<float> y,const std::atomic_bool& cancel) override {
  ++attention_calls;if(fail||cancel.load())throw std::runtime_error("test_compute_failure");
  if(q.size()!=rows*heads*dim||k.size()!=keys*kv*dim||v.size()!=k.size()||y.size()!=q.size())throw std::runtime_error("test_attention_shape");
  std::vector<float> scores(keys);
  for(std::size_t t=0;t<rows;++t)for(std::size_t h=0;h<heads;++h){float maximum=-INFINITY;const auto kh=h/(heads/kv);
   for(std::size_t s=0;s<keys;++s){float dot=0;for(std::size_t c=0;c<dim;++c)dot+=q[(t*heads+h)*dim+c]*k[(s*kv+kh)*dim+c];scores[s]=kadan::attention_visible(t,s,causal,window,offset)?dot/std::sqrt(float(dim)):-INFINITY;maximum=std::max(maximum,scores[s]);}
   double sum=0;for(float& score:scores){score=std::exp(score-maximum);sum+=score;}if(!(sum>0))throw std::runtime_error("test_attention_mask");
   for(std::size_t c=0;c<dim;++c){float value=0;for(std::size_t s=0;s<keys;++s)value+=float(scores[s]/sum)*v[(s*kv+kh)*dim+c];y[(t*heads+h)*dim+c]=value;}
  }return true;
 }
 void dense(std::span<const float> w,std::span<const float> b,std::span<const float> x,std::size_t in,std::size_t out,std::span<float> y,const std::atomic_bool& cancel,bool) override {
  if(cancel.load())throw std::runtime_error("test_compute_cancelled");
  if(!in||!out||x.size()%in||w.size()!=in*out||y.size()!=x.size()/in*out||(!b.empty()&&b.size()!=out))throw std::runtime_error("test_compute_shape");
  ++calls;if(fail)throw std::runtime_error("test_compute_failure");
  for(std::size_t row=0;row<x.size()/in;++row)for(std::size_t o=0;o<out;++o){float v=b.empty()?0:b[o];for(std::size_t c=0;c<in;++c)v+=x[row*in+c]*w[o*in+c];y[row*out+o]=v;}
 }
};
inline std::shared_ptr<RecordingDense> test_dense(){return std::getenv("KADAN_TEST_DENSE")?std::make_shared<RecordingDense>():nullptr;}
inline void check_dense(const std::shared_ptr<RecordingDense>& c){if(c){if(!c->calls||!c->attention_calls)throw std::runtime_error("compute_dispatch_missing");std::cout<<"CPU projection dispatch calls="<<c->calls<<" attention calls="<<c->attention_calls<<'\n';}}
