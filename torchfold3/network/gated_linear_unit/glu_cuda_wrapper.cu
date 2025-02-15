// glu_cuda.cpp

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// 声明CUDA内核模板（上面已经定义）
template <typename T, typename acc_t>
__global__ void glu_kernel(
    const T* __restrict__ x,
    const T* __restrict__ w_gate,
    const T* __restrict__ w_proj,
    T* __restrict__ out,
    int M, int N, int K,
    int stride_x0, int stride_x1,
    int stride_wg0, int stride_wg1,
    int stride_wp0, int stride_wp1,
    int stride_out0, int stride_out1);

// 封装接口
torch::Tensor glu_cuda(torch::Tensor x, torch::Tensor weight) {

    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    int M = x_contig.size(0);
    int K = x_contig.size(1);
    TORCH_CHECK(weight_contig.size(0) == K, "x and weight dim mismatch");
    int twoN = weight_contig.size(1);
    TORCH_CHECK(twoN % 2 == 0, "weight's second dim must be even");
    int N = twoN / 2;

    // 输出张量
    auto out = torch::empty({M, N}, x_contig.options());

    // 设置CUDA kernel的线程块与网格尺寸：
    // 每个线程块处理一个128x128的tile，故：
    dim3 block(32, 8);  // 32×8=256个线程
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);

    // 获取stride信息（注意：这里stride单位为元素，不是字节）
    int stride_x0 = x_contig.stride(0);
    int stride_x1 = x_contig.stride(1);
    int stride_wg0 = weight_contig.stride(0);  // w_gate部分：实际weight的stride[0]为 2*N
    int stride_wg1 = weight_contig.stride(1);
    int stride_wp0 = weight_contig.stride(0);  // w_proj部分，同样stride
    int stride_wp1 = weight_contig.stride(1);
    int stride_out0 = out.stride(0);
    int stride_out1 = out.stride(1);

    // 对输入数据类型进行dispatch，支持float和bfloat16，累加统一用float
    AT_DISPATCH_FLOATING_TYPES_AND(at::BFloat16, x_contig.scalar_type(), "glu_cuda", ([&] {
        using scalar_t = scalar_t;
        using acc_t = float;  // 无论输入是什么类型，accumulation均用float
        // 注意：w_gate指针为weight_contig.data_ptr<scalar_t>()（前N列），
        //       w_proj指针为weight_contig.data_ptr<scalar_t>() + N（从第N列开始）
        glu_kernel<scalar_t, acc_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                x_contig.data_ptr<scalar_t>(),
                weight_contig.data_ptr<scalar_t>(),             // w_gate: 前半部分
                weight_contig.data_ptr<scalar_t>() + N,           // w_proj: 后半部分
                out.data_ptr<scalar_t>(),
                M, N, K,
                stride_x0, stride_x1,
                stride_wg0, stride_wg1,
                stride_wp0, stride_wp1,
                stride_out0, stride_out1
            );
    }));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("glu", &glu_cuda, "Gated Linear Unit CUDA implementation");
}
