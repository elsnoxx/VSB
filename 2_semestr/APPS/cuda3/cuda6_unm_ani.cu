#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_img.h"
#include "animation.h"

__global__ void kernel_insert_rgb_image(CudaImg t_big_rgba_pic, CudaImg t_small_rgb_pic, int pos_x, int pos_y, uint8_t t_alpha)
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_small_rgb_pic.m_size.y || l_x >= t_small_rgb_pic.m_size.x)
        return;

    int l_by = l_y + pos_y;
    int l_bx = l_x + pos_x;
    if (l_by >= t_big_rgba_pic.m_size.y || l_by < 0 || l_bx >= t_big_rgba_pic.m_size.x || l_bx < 0)
        return;

    // Get pixel from small RGB image
    uchar3 l_rgb = t_small_rgb_pic.m_p_uchar3[l_y * t_small_rgb_pic.m_size.x + l_x];

    // Apply alpha blending based on provided alpha value
    float alpha = (float)t_alpha / 255.0f;
    uchar4 l_rgba;
    l_rgba.x = l_rgb.x;
    l_rgba.y = l_rgb.y;
    l_rgba.z = l_rgb.z;
    l_rgba.w = t_alpha;

    // Blend with background pixel using alpha channel
    uchar4 l_bg_rgba = t_big_rgba_pic.m_p_uchar4[l_by * t_big_rgba_pic.m_size.x + l_bx];
    float bg_alpha = (float)l_bg_rgba.w / 255.0f;
    l_rgba.x = l_rgb.x * alpha + l_bg_rgba.x * (1.0f - alpha * bg_alpha);
    l_rgba.y = l_rgb.y * alpha + l_bg_rgba.y * (1.0f - alpha * bg_alpha);
    l_rgba.z = l_rgb.z * alpha + l_bg_rgba.z * (1.0f - alpha * bg_alpha);
    l_rgba.w = l_bg_rgba.w; // Preserve background alpha channel

    // Store blended pixel into image
    t_big_rgba_pic.m_p_uchar4[l_by * t_big_rgba_pic.m_size.x + l_bx] = l_rgba;
}



void cu_insert_rgb_image(CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int pos_x, int pos_y, uint8_t t_alpha)
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 32;
    dim3 l_blocks((t_small_cuda_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_small_cuda_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_insert_rgb_image<<<l_blocks, l_threads>>>(t_big_cuda_pic, t_small_cuda_pic, pos_x, pos_y, t_alpha);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_insertimage(CudaImg t_big_cuda_img, CudaImg t_small_cuda_pic, int pos_x, int pos_y, int t_alpha)
{
    // X,Y coordinates and check image dimensions
	int l_y = blockDim.y * blockIdx.y + threadIdx.y;
	int l_x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( l_y >= t_small_cuda_pic.m_size.y ) return;
	if ( l_x >= t_small_cuda_pic.m_size.x ) return;
	int l_by = l_y + pos_y;
	int l_bx = l_x + pos_x;
	if ( l_by >= t_big_cuda_img.m_size.y || l_by < 0 ) return;
	if ( l_bx >= t_big_cuda_img.m_size.x || l_bx < 0 ) return;

	// Get point from small image
	uchar4 l_fg_bgra = t_small_cuda_pic.m_p_uchar4[ l_y * t_small_cuda_pic.m_size.x + l_x ];
	uchar3 l_bg_bgr = t_big_cuda_img.m_p_uchar3[ l_by * t_big_cuda_img.m_size.x + l_bx ];
	uchar3 l_bgr = { 0, 0, 0};

	// compose point from small and big image according alpha channel
	l_bgr.x = l_fg_bgra.x * l_fg_bgra.w / t_alpha + l_bg_bgr.x * ( 255 - l_fg_bgra.w ) / t_alpha;
	l_bgr.y = l_fg_bgra.y * l_fg_bgra.w / t_alpha + l_bg_bgr.y * ( 255 - l_fg_bgra.w ) / t_alpha;
	l_bgr.z = l_fg_bgra.z * l_fg_bgra.w / t_alpha + l_bg_bgr.z * ( 255 - l_fg_bgra.w ) / t_alpha;

	// Store point into image
	t_big_cuda_img.m_p_uchar3[ l_by * t_big_cuda_img.m_size.x + l_bx ] = l_bgr;
}

void cu_insertimage(CudaImg t_big_cuda_img, CudaImg t_small_cuda_pic, int pos_x, int pos_y, int t_alpha)
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 32;
    dim3 l_blocks((t_small_cuda_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_small_cuda_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_insertimage<<<l_blocks, l_threads>>>(t_big_cuda_img, t_small_cuda_pic, pos_x, pos_y, t_alpha);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

// rotate
__global__ void kernel_rotate_image(CudaImg t_orig, CudaImg t_rotate, float t_sin, float t_cos)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= t_rotate.m_size.x || y >= t_rotate.m_size.y)
        return;

    int l_crotate_x = x - t_rotate.m_size.x / 2;
    int l_crotate_y = y - t_rotate.m_size.y / 2;

    float l_corig_x = t_cos * l_crotate_x - t_sin * l_crotate_y;
    float l_corig_y = t_sin * l_crotate_x + t_cos * l_crotate_y;

    int l_orig_x = l_corig_x + t_orig.m_size.x / 2;
    int l_orig_y = l_corig_y + t_orig.m_size.y / 2;

    if (l_orig_x >= 0 && l_orig_x < t_orig.m_size.x && l_orig_y >= 0 && l_orig_y < t_orig.m_size.y)
    {
        uchar4 l_pixel = t_orig.m_p_uchar4[l_orig_y * t_orig.m_size.x + l_orig_x];
        t_rotate.m_p_uchar4[y * t_rotate.m_size.x + x] = l_pixel;
    }
    else
    {
        t_rotate.m_p_uchar4[y * t_rotate.m_size.x + x] = make_uchar4(0, 0, 0, 0);
    }
}

void cu_run_rotate(CudaImg &t_orig, CudaImg &t_rotate, float t_angle)
{
    float t_sin = sinf(t_angle);
    float t_cos = cosf(t_angle);

    dim3 blockSize(16, 16);
    dim3 gridSize((t_rotate.m_size.x + blockSize.x - 1) / blockSize.x,
                  (t_rotate.m_size.y + blockSize.y - 1) / blockSize.y);

    kernel_rotate_image<<<gridSize, blockSize>>>(t_orig, t_rotate, t_sin, t_cos);

    cudaError_t l_cerr;
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

__global__ void kernel_mirror(CudaImg t_img_cuda)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;

    if (l_y >= t_img_cuda.m_size.y || l_x >= t_img_cuda.m_size.x)
        return;

    // Vertikální zrcadlení: výpočet pouze první poloviny obrázku
    int new_x = t_img_cuda.m_size.x - 1 - l_x;

    // Čtení a výměna pixelů pro RGBA
    uchar4 temp = t_img_cuda.m_p_uchar4[l_y * t_img_cuda.m_size.x + l_x];
    t_img_cuda.m_p_uchar4[l_y * t_img_cuda.m_size.x + l_x] = t_img_cuda.m_p_uchar4[l_y * t_img_cuda.m_size.x + new_x];
    t_img_cuda.m_p_uchar4[l_y * t_img_cuda.m_size.x + new_x] = temp;
}


void cu_run_mirror(CudaImg t_img_cuda)
{
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((t_img_cuda.m_size.x + l_block_size - 1) / l_block_size, (t_img_cuda.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    // Vertikální zrcadlení: výpočet pouze první poloviny obrázku
    l_blocks.x = (t_img_cuda.m_size.x + l_block_size - 1) / (2 * l_block_size);
    kernel_mirror<<<l_blocks, l_threads>>>(t_img_cuda);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}




// Demo kernel to create gradient
__global__ void kernel_creategradient(CudaImg t_color_cuda_img)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_color_cuda_img.m_size.y) return;
    if (l_x >= t_color_cuda_img.m_size.x) return;

    int l_dy = l_x * t_color_cuda_img.m_size.y / t_color_cuda_img.m_size.x + l_y - t_color_cuda_img.m_size.y;
    unsigned char l_color = 255 * abs(l_dy) / t_color_cuda_img.m_size.y;

    uchar3 l_bgr = (l_dy < 0) ? (uchar3){l_color, 255 - l_color, 0} : (uchar3){0, 255 - l_color, l_color};

    t_color_cuda_img.m_p_uchar3[l_y * t_color_cuda_img.m_size.x + l_x] = l_bgr;
}

// Kernel to insert image with alpha channel gradient
__global__ void kernel_insertimage(CudaImg t_big_cuda_img, CudaImg t_small_cuda_pic, int2 t_position)
{
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (l_y >= t_small_cuda_pic.m_size.y) return;
    if (l_x >= t_small_cuda_pic.m_size.x) return;
    int l_by = l_y + t_position.y;
    int l_bx = l_x + t_position.x;
    if (l_by >= t_big_cuda_img.m_size.y || l_by < 0) return;
    if (l_bx >= t_big_cuda_img.m_size.x || l_bx < 0) return;

    uchar4 l_fg_bgra = t_small_cuda_pic.m_p_uchar4[l_y * t_small_cuda_pic.m_size.x + l_x];
    uchar3 l_bg_bgr = t_big_cuda_img.m_p_uchar3[l_by * t_big_cuda_img.m_size.x + l_bx];
    uchar3 l_bgr = {0, 0, 0};

    l_bgr.x = l_fg_bgra.x * l_fg_bgra.w / 255 + l_bg_bgr.x * (255 - l_fg_bgra.w) / 255;
    l_bgr.y = l_fg_bgra.y * l_fg_bgra.w / 255 + l_bg_bgr.y * (255 - l_fg_bgra.w) / 255;
    l_bgr.z = l_fg_bgra.z * l_fg_bgra.w / 255 + l_bg_bgr.z * (255 - l_fg_bgra.w) / 255;

    t_big_cuda_img.m_p_uchar3[l_by * t_big_cuda_img.m_size.x + l_bx] = l_bgr;
}

void cu_insertimage(CudaImg t_big_cuda_img, CudaImg t_small_cuda_pic, int2 t_position)
{
    cudaError_t l_cerr;

    int l_block_size = 32;
    dim3 l_blocks((t_small_cuda_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (t_small_cuda_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_insertimage<<<l_blocks, l_threads>>>(t_big_cuda_img, t_small_cuda_pic, t_position);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

void Animation::start(CudaImg t_bg_cuda_img, CudaImg t_ins_cuda_img)
{
    if (m_initialized) return;
    cudaError_t l_cerr;

    m_bg_cuda_img = t_bg_cuda_img;
    m_res_cuda_img = t_bg_cuda_img;
    m_ins_cuda_img = t_ins_cuda_img;

    l_cerr = cudaMalloc(&m_bg_cuda_img.m_p_void, m_bg_cuda_img.m_size.x * m_bg_cuda_img.m_size.y * sizeof(uchar3));
    if (l_cerr != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    int l_block_size = 32;
    dim3 l_blocks((m_bg_cuda_img.m_size.x + l_block_size - 1) / l_block_size,
                  (m_bg_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_creategradient<<<l_blocks, l_threads>>>(m_bg_cuda_img);

    m_initialized = 1;
}

void Animation::next(CudaImg t_res_cuda_img, int2 t_position)
{
    if (!m_initialized) return;

    cudaError_t cerr;

    cerr = cudaMemcpy(m_res_cuda_img.m_p_void, m_bg_cuda_img.m_p_void, m_bg_cuda_img.m_size.x * m_bg_cuda_img.m_size.y * sizeof(uchar3), cudaMemcpyDeviceToDevice);
    if (cerr != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

    int l_block_size = 32;
    dim3 l_blocks((m_ins_cuda_img.m_size.x + l_block_size - 1) / l_block_size,
                  (m_ins_cuda_img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_insertimage<<<l_blocks, l_threads>>>(m_res_cuda_img, m_ins_cuda_img, t_position);

    cerr = cudaMemcpy(t_res_cuda_img.m_p_void, m_res_cuda_img.m_p_void, m_res_cuda_img.m_size.x * m_res_cuda_img.m_size.y * sizeof(uchar3), cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));
}

void Animation::stop()
{
    if (!m_initialized) return;

    cudaFree(m_bg_cuda_img.m_p_void);
    cudaFree(m_res_cuda_img.m_p_void);
    cudaFree(m_ins_cuda_img.m_p_void);

    m_initialized = 0;
}
