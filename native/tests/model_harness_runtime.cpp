#define KADAN_MODEL_RUNTIME
#include "stack_runtime.cpp"
#include <unistd.h>
cudaError_t cudaSetDevice(int device){return device==0?cudaSuccess:cudaErrorUnknown;}
cudaError_t cudaMemGetInfo(std::size_t*free,std::size_t*total){
    // Test-only stalled runtime: verify the production harness's watchdog exits
    // once without recovery/retry. No spin loop, GPU or service interaction.
    if(std::getenv("KADAN_TEST_STALLED_RUNTIME"))for(;;)::pause();
    *free=*total=SIZE_MAX;return cudaSuccess;
}
