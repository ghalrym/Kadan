// Kadan links no ggml backend. ggml.c and gguf.cpp reference these only for
// tensors held in backend buffers (graph gradients, GGUF writing), which Kadan
// never creates.
#include "ggml-backend.h"
#include "ggml-impl.h"

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(tensor); GGML_UNUSED(data); GGML_UNUSED(offset); GGML_UNUSED(size);
    GGML_ABORT("ggml backend is not linked");
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(tensor); GGML_UNUSED(data); GGML_UNUSED(offset); GGML_UNUSED(size);
    GGML_ABORT("ggml backend is not linked");
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(tensor); GGML_UNUSED(value); GGML_UNUSED(offset); GGML_UNUSED(size);
    GGML_ABORT("ggml backend is not linked");
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    GGML_ABORT("ggml backend is not linked");
}
