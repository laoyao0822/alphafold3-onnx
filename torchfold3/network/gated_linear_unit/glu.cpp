#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA forward declaration
void gated_linear_unit_cuda(const float* x, const float* weight, float* out, int batch_size, int m, int n, int k);

torch::Tensor gated_linear_unit_forward(torch::Tensor x, torch::Tensor weight) {
    // Check inputs
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    // Get dimensions
    int batch_size = 1;
    for (int i = 0; i < x.dim() - 2; ++i) {
        batch_size *= x.size(i);
    }
    int m = x.size(-2);
    int n = x.size(-1);
    int k = weight.size(-1) / 2;

    // Create output tensor
    auto out = torch::empty({batch_size, m, k}, x.options());

    // Call CUDA function
    gated_linear_unit_cuda(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, m, n, k
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("glu", &gated_linear_unit_forward, "Gated Linear Unit forward (CUDA)");
}