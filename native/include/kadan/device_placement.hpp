#pragma once
#include "kadan/resources.hpp"
#include <charconv>
#include <algorithm>
#include <span>
#include <string_view>
namespace kadan {
// Budgets are indexed by physical device, independent of selection order.
inline Footprint device_budgets(std::span<const int> devices,std::string_view scalar,std::string_view indexed,Bytes minimum){
    Footprint result(3);std::vector<bool> seen(3);
    auto parse=[&](std::string_view value){Bytes bytes=0;auto p=std::from_chars(value.data(),value.data()+value.size(),bytes);
        if(value.empty()||p.ec!=std::errc{}||p.ptr!=value.data()+value.size()||bytes<minimum||bytes>24ULL*1024*1024*1024)throw std::invalid_argument("device_budget");
        return bytes;};
    if(indexed.empty()){auto bytes=parse(scalar);for(int d:devices)result.at(d+1)=bytes;return result;}
    while(!indexed.empty()){
        const auto comma=indexed.find(',');const auto item=indexed.substr(0,comma);
        if(item.size()<3||item[1]!=':'||(item[0]!='0'&&item[0]!='1'))throw std::invalid_argument("device_budget_mapping");
        const int d=item[0]-'0';if(seen[d+1]||std::find(devices.begin(),devices.end(),d)==devices.end())throw std::invalid_argument("device_budget_mapping");
        seen[d+1]=true;result[d+1]=parse(item.substr(2));
        if(comma==std::string_view::npos)break;
        indexed.remove_prefix(comma+1);if(indexed.empty())throw std::invalid_argument("device_budget_mapping");
    }
    for(int d:devices)if(!seen.at(d+1))throw std::invalid_argument("device_budget_mapping");
    return result;
}
// Retry only a confirmed lack of memory; other device failures must propagate.
// The owner keeps its admission for the entire wait. Pause is bounded by caller.
template<class Stop,class Attempt,class Pause>
void wait_for_device_allocation(Stop stop,Attempt attempt,Pause pause){
    for(;;){if(stop())throw std::runtime_error("h3_cuda_cancelled");if(attempt())return;pause();}
}
// Worst-case streaming scratch, including F64 widening and 256 activation rows.
// Fix this independently of request token count so retained slice keys are stable.
inline std::size_t projection_columns(std::size_t in,std::size_t out,std::size_t width,bool precise,std::size_t alignment){
    if(!in||in>2147483647ULL||!out||(width!=1&&width!=4)||(alignment!=1&&alignment!=4)||out%alignment)throw std::invalid_argument("projection_layout");
    constexpr Bytes limit=2ULL*1024*1024*1024,tile=256;
    const Bytes fixed=8*1024*1024+4096+tile*in*(9+(precise?8:0))+tile*4;
    const Bytes per_column=in*(width+(precise?8:0))+8+tile*(8+(precise?8:0));
    if(fixed>=limit)throw std::invalid_argument("projection_scratch_limit");
    const auto columns=std::min<Bytes>(out,(limit-fixed)/per_column)/alignment*alignment;
    if(!columns)throw std::invalid_argument("projection_scratch_limit");
    return columns;
}
// Complete, monotone output-channel partitions; callers pass groups of four for
// integer projections. Long double avoids overflowing units*capacity.
inline std::size_t weight_boundary(std::size_t units,std::span<const Bytes> capacities,std::size_t index){
    if(index>=capacities.size())return units;
    long double total=0,prefix=0;for(std::size_t i=0;i<capacities.size();++i){total+=capacities[i];if(i<index)prefix+=capacities[i];}
    return total?std::min(units,std::size_t(static_cast<long double>(units)*prefix/total)):units*index/capacities.size();
}
}
