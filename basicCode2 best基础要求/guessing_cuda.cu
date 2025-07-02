#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "PCFG.h" // Assuming this header is still needed for other definitions
// guessing_cuda.cu
#include <cuda_runtime.h>
#include <string> // For std::string (if needed for utility functions, though not directly in kernel)

// 生成单个Segment的Kernel，直接写入扁平化缓冲区
__global__ void GenerateSingleSegmentKernel_FlatOutput(char **values, char *output_flat, int count, int total_len_per_guess) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        char *val = values[idx]; 
        int len = 1; // 后缀长度固定为1

        // 将字符写入到扁平化缓冲区中的正确位置
        output_flat[idx * total_len_per_guess] = val[0];
        // 添加空终止符
        output_flat[idx * total_len_per_guess + len] = '\0';
    }
}

// 生成最后一个Segment的Kernel，直接写入扁平化缓冲区
__global__ void GenerateLastSegmentKernel_FlatOutput(char *prefix, char **values, char *output_flat, int count, int prefix_len, int total_len_per_guess) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        char *val = values[idx]; 
        int val_len = 1; 
        // 复制前缀到输出缓冲区的正确位置
        for (int i = 0; i < prefix_len; ++i) {
            output_flat[idx * total_len_per_guess + i] = prefix[i];
        }        
        output_flat[idx * total_len_per_guess + prefix_len] = val[0];
        output_flat[idx * total_len_per_guess + prefix_len + val_len] = '\0';
    }
}