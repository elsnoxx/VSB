#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {
}

// Function prototypes
void cu_run_split(CudaImg t_bgr_cuda_img, CudaImg l_g_cv_img, CudaImg l_b_cv_img, CudaImg l_r_cv_img);
void cu_run_mirror(CudaImg t_bgr_cuda_img, int direction);
void cu_run_rotate(CudaImg t_img_cuda, CudaImg output_img, int direction);

int main(int t_numarg, char **t_arg) {
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    if (t_numarg < 2) {
        printf("Enter picture filename!\n");
        return 1;
    }

    // Load image
    cv::Mat l_bgr_cv_img = cv::imread(t_arg[1], cv::IMREAD_COLOR);

    if (!l_bgr_cv_img.data) {
        printf("Unable to read file '%s'\n", t_arg[1]);
        return 1;
    }

    // Create empty BW image
    // cv::Mat l_g_cv_img(l_bgr_cv_img.size(), CV_8UC3);
    // cv::Mat l_b_cv_img(l_bgr_cv_img.size(), CV_8UC3);
    // cv::Mat l_r_cv_img(l_bgr_cv_img.size(), CV_8UC3);
    cv::Mat l_90r_cv_img(l_bgr_cv_img.size().width, l_bgr_cv_img.size().height, CV_8UC3);
    cv::Mat l_90l_cv_img(l_bgr_cv_img.size().width, l_bgr_cv_img.size().height, CV_8UC3);
    cv::Mat l_g_cv_img(l_bgr_cv_img.size().width, l_bgr_cv_img.size().height, CV_8UC3);
    cv::Mat l_b_cv_img(l_bgr_cv_img.size().width, l_bgr_cv_img.size().height, CV_8UC3);
    cv::Mat l_r_cv_img(l_bgr_cv_img.size().width, l_bgr_cv_img.size().height, CV_8UC3);

    // Data for CUDA
    CudaImg l_bgr_cuda_img, l_green_cuda_img, l_blue_cuda_img, l_red_cuda_img;
    l_bgr_cuda_img.m_size.x = l_bgr_cv_img.size().width;
    l_bgr_cuda_img.m_size.y = l_bgr_cv_img.size().height;

    l_green_cuda_img.m_size.x = l_blue_cuda_img.m_size.x = l_red_cuda_img.m_size.x = l_bgr_cv_img.size().height;
    l_green_cuda_img.m_size.y = l_blue_cuda_img.m_size.y = l_red_cuda_img.m_size.y = l_bgr_cv_img.size().width;

    l_bgr_cuda_img.m_p_uchar3 = (uchar3 *)l_bgr_cv_img.data;
    l_green_cuda_img.m_p_uchar3 = (uchar3 *)l_g_cv_img.data;
    l_blue_cuda_img.m_p_uchar3 = (uchar3 *)l_b_cv_img.data;
    l_red_cuda_img.m_p_uchar3 = (uchar3 *)l_r_cv_img.data;

    cv::imshow("Start", l_bgr_cv_img);
    
    cu_run_mirror(l_bgr_cuda_img, 2); // Vertical flip
    cv::imshow("Vertical", l_bgr_cv_img);
    // Flip the image horizontally and vertically
    cu_run_mirror(l_bgr_cuda_img, 1); // Horizontal flip
    cv::imshow("Horizontal", l_bgr_cv_img);
    

    // Rotate the image right by 90 degrees
    CudaImg l_r90_cuda_img;
    l_r90_cuda_img.m_size.y = l_bgr_cv_img.size().width;
    l_r90_cuda_img.m_size.x = l_bgr_cv_img.size().height;
    l_r90_cuda_img.m_p_uchar3 = (uchar3 *)l_90r_cv_img.data;
    cu_run_rotate(l_bgr_cuda_img, l_r90_cuda_img, 1);
    cv::imshow("Rotate right 90", l_90r_cv_img);

    // Rotate the image left by 90 degrees
    CudaImg l_l90_cuda_img;
    l_l90_cuda_img.m_size.y = l_bgr_cv_img.size().width;
    l_l90_cuda_img.m_size.x = l_bgr_cv_img.size().height;
    l_l90_cuda_img.m_p_uchar3 = (uchar3 *)l_90l_cv_img.data;
    cu_run_rotate(l_bgr_cuda_img, l_l90_cuda_img, 2);
    cv::imshow("Rotate left 90", l_90l_cv_img);

    // Split the image rotated to the right into its color channels
    cu_run_split(l_r90_cuda_img, l_green_cuda_img, l_blue_cuda_img, l_red_cuda_img);
    cv::imshow("Green", l_g_cv_img);
    cv::imshow("Blue", l_b_cv_img);
    cv::imshow("Red", l_r_cv_img);

    cv::waitKey(0);
}
