#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_rotateClockwise(CudaImg input_img, CudaImg output_img, int direction) {
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (direction == 1) { // Rotace o 90 stupňů doprava
        if (l_y < input_img.m_size.y && l_x < input_img.m_size.x) {
            int new_x = input_img.m_size.y - 1 - l_y;
            int new_y = l_x;
            output_img.m_p_uchar3[new_y * output_img.m_size.x + new_x] = input_img.m_p_uchar3[l_y * input_img.m_size.x + l_x];
        }
    }
    else if (direction == 2) { // Rotace o 90 stupňů doleva
        if (l_y < input_img.m_size.y && l_x < input_img.m_size.x) {
            int new_x = l_y;
            int new_y = input_img.m_size.x - 1 - l_x;
            output_img.m_p_uchar3[new_y * output_img.m_size.x + new_x] = input_img.m_p_uchar3[l_y * input_img.m_size.x + l_x];
        }
    }     
}



void cu_run_rotate(CudaImg t_img_cuda, CudaImg output_img, int direction)
{
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((t_img_cuda.m_size.x + l_block_size - 1) / l_block_size, (t_img_cuda.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_rotateClockwise<<<l_blocks, l_threads>>>(t_img_cuda, output_img, direction);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
