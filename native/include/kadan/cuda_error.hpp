#pragma once
#include <stdexcept>
namespace kadan::cuda {
// An operation could not establish that queued work stopped touching borrowed
// buffers. Keep every borrowed input/output allocation pinned and alive until
// the owning operation's close() succeeds, or its CUDA context is torn down.
// Releasing only the callee's internal reservation is not sufficient.
class DeviceBufferQuarantine : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
} // namespace kadan::cuda
