#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel for SiLU activation and element-wise multiplication
__global__ void silu_and_multiply_kernel(const float* a, const float* b, float* out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = a[idx] / (1.0f + expf(-a[idx])) * b[idx];
    }
}

// Main function to compute Gated Linear Unit
void gated_linear_unit_cuda(const float* x, const float* weight, float* out, int batch_size, int m, int n, int k) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate device memory for intermediate result y
    float* y;
    CHECK_CUDA(cudaMalloc(&y, batch_size * m * 2 * k * sizeof(float)));

    // Perform batched matrix multiplication using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        2 * k, m, n, &alpha,
        weight, 2 * k, 2 * k * n,
        x, n, m * n,
        &beta,
        y, 2 * k, m * 2 * k,
        batch_size
    ));

    // Split y into a and b
    float* a = y;
    float* b = y + batch_size * m * k;

    // Launch kernel to compute SiLU(a) * b
    int num_elements = batch_size * m * k;
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    silu_and_multiply_kernel<<<blocks_per_grid, threads_per_block>>>(a, b, out, num_elements);

    // Free device memory
    // CHECK_CUDA(cudaFree(y));
    CHECK_CUBLAS(cublasDestroy(handle));
}