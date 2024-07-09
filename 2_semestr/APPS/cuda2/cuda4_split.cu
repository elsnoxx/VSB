#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

// Demo kernel to transform RGB color schema to BW schema
__global__ void kernel_split( CudaImg t_color_cuda_img, CudaImg l_g_cv_img, CudaImg l_b_cv_img, CudaImg l_r_cv_img)
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= t_color_cuda_img.m_size.y ) return;
    if ( l_x >= t_color_cuda_img.m_size.x ) return;

    // Get point from color picture
    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[ l_y * t_color_cuda_img.m_size.x + l_x ];

    // Green img
    l_g_cv_img.m_p_uchar3[ l_y * l_g_cv_img.m_size.x + l_x ].x = 0;
    l_g_cv_img.m_p_uchar3[ l_y * l_g_cv_img.m_size.x + l_x ].y = l_bgr.y;
    l_g_cv_img.m_p_uchar3[ l_y * l_g_cv_img.m_size.x + l_x ].z = 0;

    // Blue img
    l_b_cv_img.m_p_uchar3[ l_y * l_b_cv_img.m_size.x + l_x ].x = l_bgr.x;
    l_b_cv_img.m_p_uchar3[ l_y * l_b_cv_img.m_size.x + l_x ].y = 0;
    l_b_cv_img.m_p_uchar3[ l_y * l_b_cv_img.m_size.x + l_x ].z = 0;

    // Red img
    l_r_cv_img.m_p_uchar3[ l_y * l_b_cv_img.m_size.x + l_x ].x = 0;
    l_r_cv_img.m_p_uchar3[ l_y * l_b_cv_img.m_size.x + l_x ].y = 0;
    l_r_cv_img.m_p_uchar3[ l_y * l_b_cv_img.m_size.x + l_x ].z = l_bgr.z;
}

void cu_run_split( CudaImg t_color_cuda_img, CudaImg l_g_cv_img,  CudaImg l_b_cv_img, CudaImg l_r_cv_img )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks( ( t_color_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( t_color_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    kernel_split<<< l_blocks, l_threads >>>( t_color_cuda_img, l_g_cv_img, l_b_cv_img, l_r_cv_img );

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}