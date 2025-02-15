// glu_cuda.cu

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 为任意类型定义device版sigmoid（仅用于中间计算，accumulation统一用float）
template <typename T>
__device__ __forceinline__ T sigmoid_device(T x) {
    return T(1) / (T(1) + exp(-x));
}

template <>
__device__ __forceinline__ float sigmoid_device<float>(float x) {
    return 1.f / (1.f + expf(-x));
}

template <>
__device__ __forceinline__ __nv_bfloat16 sigmoid_device<__nv_bfloat16>(__nv_bfloat16 x) {
    float xf = __bfloat162float(x);
    float res = 1.f / (1.f + expf(-xf));
    return __float2bfloat16_rn(res);
}

// CUDA内核：同时计算x*w_gate和x*w_proj，最后计算 out = acc0 * sigmoid(acc0) * acc1
template <typename T, typename acc_t>
__global__ void glu_kernel(
    const T* __restrict__ x,         // [M, K]
    const T* __restrict__ w_gate,    // [K, N]
    const T* __restrict__ w_proj,    // [K, N]
    T* __restrict__ out,             // [M, N]
    int M, int N, int K,
    int stride_x0, int stride_x1,      // x的两个维度的stride
    int stride_wg0, int stride_wg1,    // w_gate的stride
    int stride_wp0, int stride_wp1,    // w_proj的stride
    int stride_out0, int stride_out1   // out的stride
) {
    // 定义tile尺寸
    constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_N = 128;
    constexpr int BLOCK_SIZE_K = 32;

    // 计算当前线程块处理的输出tile在全局矩阵中的起始行、列
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int rowStart = blockRow * BLOCK_SIZE_M;
    int colStart = blockCol * BLOCK_SIZE_N;

    // 线程块内线程编号
    int tx = threadIdx.x;  // 横向编号
    int ty = threadIdx.y;  // 纵向编号

    // 每个线程处理子块尺寸（要求 BLOCK_SIZE_M 与 BLOCK_SIZE_N 能被blockDim分别整除）
    constexpr int numRowsPerThread = BLOCK_SIZE_M / 8;    // 例如：128/8 = 16
    constexpr int numColsPerThread = BLOCK_SIZE_N / 32;   // 例如：128/32 = 4

    // 寄存器保存子块accumulator，注意：accumulation统一用float类型（acc_t）
    acc_t acc0[numRowsPerThread][numColsPerThread];
    acc_t acc1[numRowsPerThread][numColsPerThread];
#pragma unroll
    for (int i = 0; i < numRowsPerThread; i++) {
#pragma unroll
        for (int j = 0; j < numColsPerThread; j++) {
            acc0[i][j] = acc_t(0);
            acc1[i][j] = acc_t(0);
        }
    }

    // 分配共享内存，用于存储A（x）和B（w_gate、w_proj）的tile
    __shared__ T As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ T Bgs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    __shared__ T Bps[BLOCK_SIZE_K][BLOCK_SIZE_N];

    // k方向按tile循环
    for (int kTile = 0; kTile < K; kTile += BLOCK_SIZE_K) {
        // --- 加载x的tile到共享内存 As ---
        // 每个线程加载其负责的若干行
        for (int i = 0; i < numRowsPerThread; i++) {
            int globalRow = rowStart + ty * numRowsPerThread + i;
            // 让每个线程横向加载BLOCK_SIZE_K列
            for (int j = tx; j < BLOCK_SIZE_K; j += blockDim.x) {
                int globalK = kTile + j;
                if (globalRow < M && globalK < K)
                    As[ty * numRowsPerThread + i][j] = x[globalRow * stride_x0 + globalK * stride_x1];
                else
                    As[ty * numRowsPerThread + i][j] = T(0);
            }
        }

        // --- 加载w_gate和w_proj的tile到共享内存 ---
        // 注意：w_gate和w_proj均为[K, N]，其中globalK = kTile + i, globalCol = colStart + j
        for (int i = ty; i < BLOCK_SIZE_K; i += blockDim.y) {
            int globalK = kTile + i;
            for (int j = tx; j < BLOCK_SIZE_N; j += blockDim.x) {
                int globalCol = colStart + j;
                if (globalK < K && globalCol < N) {
                    Bgs[i][j] = w_gate[globalK * stride_wg0 + globalCol * stride_wg1];
                    Bps[i][j] = w_proj[globalK * stride_wp0 + globalCol * stride_wp1];
                } else {
                    Bgs[i][j] = T(0);
                    Bps[i][j] = T(0);
                }
            }
        }
        __syncthreads();

        // --- 累加计算当前tile部分 ---
        for (int kInner = 0; kInner < BLOCK_SIZE_K; kInner++) {
            // 每个线程取出自己对应行的元素（存入寄存器aFragment）
            T aFragment[numRowsPerThread];
#pragma unroll
            for (int i = 0; i < numRowsPerThread; i++) {
                aFragment[i] = As[ty * numRowsPerThread + i][kInner];
            }
#pragma unroll
            for (int j = 0; j < numColsPerThread; j++) {
                int colIdx = tx * numColsPerThread + j;  // 对应共享内存中B的列号
                T bGate = Bgs[kInner][colIdx];
                T bProj = Bps[kInner][colIdx];
#pragma unroll
                for (int i = 0; i < numRowsPerThread; i++) {
                    acc0[i][j] += static_cast<acc_t>(aFragment[i]) * static_cast<acc_t>(bGate);
                    acc1[i][j] += static_cast<acc_t>(aFragment[i]) * static_cast<acc_t>(bProj);
                }
            }
        }
        __syncthreads();
    } // end for kTile

    // --- 将计算结果写回全局内存 ---
    for (int i = 0; i < numRowsPerThread; i++) {
        int globalRow = rowStart + ty * numRowsPerThread + i;
        if (globalRow < M) {
            for (int j = 0; j < numColsPerThread; j++) {
                int globalCol = colStart + tx * numColsPerThread + j;
                if (globalCol < N) {
                    // 计算sigmoid(acc0)（转换为float计算）
                    float sig = 1.f / (1.f + expf(-static_cast<float>(acc0[i][j])));
                    acc_t res = acc0[i][j] * static_cast<acc_t>(sig) * acc1[i][j];
                    out[globalRow * stride_out0 + globalCol * stride_out1] = static_cast<T>(res);
                }
            }
        }
    }
}
