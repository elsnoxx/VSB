#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

__global__ void kernel_mirror(CudaImg t_img_cuda, int direction)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (l_y >= (t_img_cuda.m_size.y) || l_x >= (t_img_cuda.m_size.x))
            return;
    

    if (direction == 1){
        // Horizontální zrcadlení: výpočet pouze první poloviny obrázku
        int new_y = t_img_cuda.m_size.y - 1 - l_y;

        uchar3 temp = t_img_cuda.m_p_uchar3[l_y * t_img_cuda.m_size.x + l_x];
        t_img_cuda.m_p_uchar3[l_y * t_img_cuda.m_size.x + l_x] = t_img_cuda.m_p_uchar3[new_y * t_img_cuda.m_size.x + l_x];
        t_img_cuda.m_p_uchar3[new_y * t_img_cuda.m_size.x + l_x] = temp;
    }
    else if (direction == 2){
        // Vertikální zrcadlení: výpočet pouze první poloviny obrázku
        int new_x = t_img_cuda.m_size.x - 1 - l_x;

        uchar3 temp = t_img_cuda.m_p_uchar3[l_y * t_img_cuda.m_size.x + l_x];
        t_img_cuda.m_p_uchar3[l_y * t_img_cuda.m_size.x + l_x] = t_img_cuda.m_p_uchar3[l_y * t_img_cuda.m_size.x + new_x];
        t_img_cuda.m_p_uchar3[l_y * t_img_cuda.m_size.x + new_x] = temp;
    }
    
}

void cu_run_mirror(CudaImg t_img_cuda, int direction)
{
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((t_img_cuda.m_size.x + l_block_size - 1) / l_block_size, (t_img_cuda.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    if (direction == 1) {
        // Horizontální zrcadlení: výpočet pouze první poloviny obrázku
        l_blocks.y = (t_img_cuda.m_size.y + l_block_size - 1) / (2 * l_block_size);
        kernel_mirror<<<l_blocks, l_threads>>>(t_img_cuda, direction);
    }
    else if (direction == 2) {
        // Vertikální zrcadlení: výpočet pouze první poloviny obrázku
        l_blocks.x = (t_img_cuda.m_size.x + l_block_size - 1) / (2 * l_block_size);
        kernel_mirror<<<l_blocks, l_threads>>>(t_img_cuda, direction);
    }

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
