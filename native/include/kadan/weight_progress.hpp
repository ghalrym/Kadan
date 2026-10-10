#pragma once
#include "kadan/dense_compute.hpp"
#include "kadan/weight_inventory.hpp"
namespace kadan {
// Existing stage/value protocol, exact bytes. Emit only at model/coarse request phase boundaries.
// Inventory covers checkpoint tensors, not measured residency or a retention limit.
template<class Hook> void report_weight_inventory(const checkpoint::WeightInventory& inventory,const Hook& hook){
    if(!hook)return;
    hook("checkpoint_tensor_bytes",inventory.stored_bytes);
    hook("expanded_f32_inventory_bytes",inventory.floating_f32_bytes);
    hook("nonfloating_inventory_bytes",inventory.raw_nonfloating_bytes);
}
template<class Hook> void report_weight_placement(const DenseCompute& compute,const Hook& hook){
    if(!hook)return;
    for(const auto& device:compute.weight_placement()){
        if(device.device!=0&&device.device!=1)throw std::invalid_argument("weight_progress_device");
        hook(device.device==0?"gpu_0_weight_planned_bytes":"gpu_1_weight_planned_bytes",device.planned_bytes);
        hook(device.device==0?"gpu_0_weight_allocated_bytes":"gpu_1_weight_allocated_bytes",device.allocated_bytes);
    }
}
}
