#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "uni_mem_allocator.h"
#include "cuda_img.h"
#include "animation.h"

extern void cu_insert_rgb_image(CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int pos_x, int pos_y, uint8_t t_alpha);
extern void cu_insertimage(CudaImg t_big_cuda_img, CudaImg t_small_cuda_pic, int pos_x, int pos_y, int t_alpha);
extern void cu_run_rotate(CudaImg &t_orig, CudaImg &t_rotate, float t_angle);
extern void cu_run_mirror(CudaImg t_img_cuda);

void set_cuda_img(cv::Mat &t_cv_img, CudaImg &t_cuda_img)
{
    t_cuda_img.m_size.x = t_cv_img.cols;
    t_cuda_img.m_size.y = t_cv_img.rows;
    t_cuda_img.channels = t_cv_img.channels();
    t_cuda_img.m_size.z = t_cuda_img.channels;
    t_cuda_img.m_p_void = t_cv_img.data;
    // printf("%d\n",t_cuda_img.channels);
    switch (t_cuda_img.channels)
    {
    case 1:
        t_cuda_img.m_p_uchar1 = reinterpret_cast<uchar1 *>(t_cv_img.data);
        break;
    case 3:
        t_cuda_img.m_p_uchar3 = reinterpret_cast<uchar3 *>(t_cv_img.data);
        break;
    case 4:
        t_cuda_img.m_p_uchar4 = reinterpret_cast<uchar4 *>(t_cv_img.data);
        break;
    default:
        t_cuda_img.m_p_void = nullptr;
        break;
    }
}


void animate_mood_change(cv::Mat &background, cv::Mat &smile_pos, cv::Mat &smile_neg)
{
    CudaImg cuda_background, cuda_smile_pos, cuda_smile_neg, cuda_result;
    set_cuda_img(background, cuda_background);
    set_cuda_img(smile_pos, cuda_smile_pos);
    set_cuda_img(smile_neg, cuda_smile_neg);

    cv::Mat result = background.clone();
    set_cuda_img(result, cuda_result);

    // Initialize video writer
    cv::VideoWriter l_video("mood_change.mkv", cv::VideoWriter::fourcc('H', '2', '6', '4'), 1, background.size());

    for (uint8_t alpha_level = 0; alpha_level <= 255; ++alpha_level)
    {
        result = background.clone(); // Reset the result image to the original background
        set_cuda_img(result, cuda_result);

        // Insert smile_neg with varying alpha levels
        cu_insert_rgb_image(cuda_result, cuda_smile_pos, 0, 0, 255 - alpha_level);
        // Insert smile_pos with varying alpha levels
        cu_insert_rgb_image(cuda_result, cuda_smile_neg, 0, 0, alpha_level);
        // Write the frame to the video
        l_video.write(result);

        // Show the result
        cv::imshow("Mood Change Animation", result);
        cv::waitKey(5);
        if (alpha_level == 255)
        {
            break;
        }
        
    }
    // Release the video writer
    l_video.release();
    cv::waitKey(0); 
}


void animate_rotation(cv::Mat &background, cv::Mat &orig_img, float start_angle, float end_angle, float step, int pos_x, int pos_y)
{
    CudaImg cuda_background, cuda_orig_img, cuda_rotate_img, cuda_rotate_img2, cuda_rotate_img3, cuda_result;
    cv::Mat rotate_img(orig_img.size(), orig_img.type());
    set_cuda_img(background, cuda_background);
    set_cuda_img(orig_img, cuda_orig_img);
    set_cuda_img(rotate_img, cuda_rotate_img);
    set_cuda_img(rotate_img, cuda_rotate_img2);
    set_cuda_img(rotate_img, cuda_rotate_img3);

    cv::Mat result = background.clone();
    set_cuda_img(result, cuda_result);

    // Initialize video writer
    cv::VideoWriter l_video("rotation.mkv", cv::VideoWriter::fourcc('H', '2', '6', '4'), 25, background.size());

    for (float angle = start_angle; angle <= end_angle; angle += step)
    {
        result = background.clone(); // Reset the result image to the original background
        set_cuda_img(result, cuda_result);

        // Rotate the first image at the given angle
        cu_run_rotate(cuda_orig_img, cuda_rotate_img, angle);
        cu_insertimage(cuda_result, cuda_rotate_img, pos_x, pos_y, 255);

        // Rotate the second image three times faster
        cu_run_rotate(cuda_orig_img, cuda_rotate_img2, angle * 3);
        cu_insertimage(cuda_result, cuda_rotate_img2, pos_x + 400, pos_y + 200, 255);

        // Rotate the third image at the same speed as the first one
        cu_run_rotate(cuda_rotate_img, cuda_rotate_img3, angle);
        cu_insertimage(cuda_result, cuda_rotate_img3, pos_x + 700, pos_y + 400, 255);

        // Display the result
        cv::imshow("Rotation Animation", result);
        
        // Write the frame to the video
        l_video.write(result);

        cv::waitKey(1); // Adjust the delay to control the speed of the animation
    }

    // Release the video writer
    l_video.release();

    cv::waitKey(0); // Wait indefinitely until a key is pressed
}

void animate_rotation_task(cv::Mat &background, cv::Mat &orig_img, cv::Mat &orig_img2, float start_angle, float end_angle, float step, int pos_x, int pos_y)
{
    CudaImg cuda_background, cuda_orig_img, cuda_orig_img3, cuda_rotate_img, cuda_rotate_img2, cuda_rotate_img3, cuda_result;
    cv::Mat rotate_img(orig_img.size(), orig_img.type());
    cv::Mat rotate_img2(orig_img2.size(), orig_img2.type());
    set_cuda_img(background, cuda_background);
    set_cuda_img(orig_img, cuda_orig_img);
    set_cuda_img(rotate_img, cuda_rotate_img);
    set_cuda_img(orig_img2, cuda_orig_img3);
    set_cuda_img(rotate_img2, cuda_rotate_img2);
    

    cv::Mat result = background.clone();
    set_cuda_img(result, cuda_result);

    // Initialize video writer
    //cv::VideoWriter l_video("rotation.mkv", cv::VideoWriter::fourcc('H', '2', '6', '4'), 25, background.size());

    for (float angle = start_angle; angle <= end_angle; angle += step)
    {
        result = background.clone(); // Reset the result image to the original background
        set_cuda_img(result, cuda_result);

        // Rotate the first image at the given angle
        //cu_run_rotate(cuda_orig_img, cuda_rotate_img, angle);
        //cu_insertimage(cuda_result, cuda_rotate_img, pos_x, pos_y, 255);

        // Rotate the second image three times faster
        
        cu_insertimage(cuda_result, cuda_orig_img, (background.cols / 2) - orig_img.cols / 2, (background.rows / 2) - orig_img.rows / 2, 255);

        float t_sin = sinf(angle);
        float t_cos = cosf(angle);
        printf("sin %f, cos %f\n", t_sin, t_cos);
        //cu_run_rotate(cuda_orig_img, cuda_rotate_img2, 90 );
        cu_run_rotate(cuda_orig_img3, cuda_rotate_img2, angle * 2);
        //cu_insertimage(cuda_result, cuda_rotate_img2, t_sin * orig_img.cols + pos_x + 400, t_cos * orig_img.cols + pos_y + 200, 255);
        cu_insertimage(cuda_result, cuda_rotate_img2, t_sin * orig_img.cols + (background.cols / 2) - orig_img.cols / 2, t_cos * orig_img.cols + (background.rows / 2) - orig_img.rows / 2, 255);
        // Rotate the third image at the same cuda_rotate_img3 as the first one
        //cu_run_rotate(cuda_rotate_img, cuda_rotate_img3, angle);
        //cu_insertimage(cuda_result, cuda_rotate_img3, pos_x + 700, pos_y + 400, 255);

        // Display the result
        cv::imshow("Rotation Animation", result);
        
        // Write the frame to the video
        //l_video.write(result);

        cv::waitKey(50); // Adjust the delay to control the speed of the animation
    }

    // Release the video writer
    //l_video.release();

    cv::waitKey(0); // Wait indefinitely until a key is pressed
}


void animate_watchtig(cv::Mat &background, std::vector<cv::Mat> &tiger_imgs, int fps, int speed)
{
    CudaImg cuda_background, cuda_tiger_img, cuda_result;
    set_cuda_img(background, cuda_background);

    cv::Mat result = background.clone();
    set_cuda_img(result, cuda_result);

    int num_imgs = tiger_imgs.size();
    int max_x = background.cols - tiger_imgs[0].cols;
    int pos_y = 400; // Fixed y-position for inserting tiger

    bool moving_right = true;
    int pos_x = -tiger_imgs[0].cols; // Start from the left side of the screen
    int frame = 0;

    // Initialize video writer
    cv::VideoWriter l_video("video.mkv", cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, background.size());

    while (true)
    {
        result = background.clone(); // Reset the result image to the original background
        set_cuda_img(result, cuda_result);

        int idx = frame % num_imgs; // Index of the current tiger image
        pos_x = moving_right ? pos_x + speed : pos_x - speed;

        set_cuda_img(tiger_imgs[idx], cuda_tiger_img);
        cu_insertimage(cuda_result, cuda_tiger_img, pos_x, pos_y, 255);
        printf("x: %d, y: %d\n", pos_x, pos_y);

        cv::imshow("WatchTig Animation", result);
        
        // Write the frame to the video
        l_video.write(result);

        // Calculate delay for given FPS
        int delay = 1000 / fps;
        cv::waitKey(delay);

        // Check if the tiger has moved out of the visible area
        if (moving_right && pos_x > background.cols)
        {
            moving_right = false;
            for (auto& img : tiger_imgs)
            {
                CudaImg cuda_img;
                set_cuda_img(img, cuda_img);
                cu_run_mirror(cuda_img);
            }
        }
        else if (!moving_right && pos_x < -tiger_imgs[0].cols)
        {
            moving_right = true;
            for (auto& img : tiger_imgs)
            {
                CudaImg cuda_img;
                set_cuda_img(img, cuda_img);
                cu_run_mirror(cuda_img);
            }
        }

        ++frame;
    }

    // Release the video writer
    l_video.release();

    // Wait for user to close the window
    cv::waitKey(0);
}




int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <1 for mood change animation, 2 for rotation animation, 3 for WatchTig animation>" << std::endl;
        return -1;
    }

    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    int vyber = atoi(argv[1]);

    switch (vyber)
    {
    case 1:
    {
        cv::Mat background(400, 400, CV_8UC4);
        cv::Mat smile_pos = cv::imread("img/smile/smile-pos.jpg", cv::IMREAD_COLOR);
        cv::Mat smile_neg = cv::imread("img/smile/smile-neg.jpg", cv::IMREAD_COLOR);

        if (smile_pos.empty() || smile_neg.empty())
        {
            std::cerr << "Images could not be loaded!" << std::endl;
            return -1;
        }

        animate_mood_change(background, smile_pos, smile_neg);
        break;
    }
    case 2:
    {
        cv::Mat background = cv::imread("img/windmill/louka.jpg", cv::IMREAD_UNCHANGED);
        cv::Mat orig_img = cv::imread("img/windmill/windmill.png", cv::IMREAD_UNCHANGED);

        if (orig_img.empty())
        {
            std::cerr << "Image could not be loaded!" << std::endl;
            return -1;
        }

        // Zvolené pozice pro vložení otáčejícího se obrázku
        int pos_x = 10;
        int pos_y = 10;

        animate_rotation(background, orig_img, 0, 100 * CV_PI, 0.1, pos_x, pos_y);
        break;
    }
    case 3:
    {
        cv::Mat background = cv::imread("img/tiger/vsbfei.jpg", cv::IMREAD_UNCHANGED);
        std::vector<cv::Mat> tiger_imgs;

        for (int i = 1; i <= 15; ++i)
        {
            std::string filename = "img/tiger/tiger";
            filename += (i < 10) ? "0" + std::to_string(i) : std::to_string(i);
            filename += ".png";
            cv::Mat tiger_img = cv::imread(filename, cv::IMREAD_UNCHANGED);
            if (tiger_img.empty())
            {
                std::cerr << "Image " << filename << " could not be loaded!" << std::endl;
                return -1;
            }
            tiger_imgs.push_back(tiger_img);

            // Debug output for image dimensions
            std::cout << "Loaded tiger image: " << filename << " with dimensions " << tiger_img.cols << "x" << tiger_img.rows << " chanels " << tiger_img.channels()<< std::endl;
        }

        // Debug output for background dimensions
        std::cout << "Loaded background image: vsbfei.jpeg with dimensions " << background.cols << "x" << background.rows << std::endl;

        animate_watchtig(background, tiger_imgs, 15, 15);

        break;
    }
    case 4:
    {
        cv::Mat background = cv::imread("img/windmill/louka.jpg", cv::IMREAD_UNCHANGED);
        cv::Mat orig_img = cv::imread("img/windmill/windmill.png", cv::IMREAD_UNCHANGED);
        cv::Mat orig_img2 = cv::imread("img/windmill/windmill.png", cv::IMREAD_UNCHANGED);

        if (orig_img.empty())
        {
            std::cerr << "Image could not be loaded!" << std::endl;
            return -1;
        }

        // Zvolené pozice pro vložení otáčejícího se obrázku
        int pos_x = 10;
        int pos_y = 10;

        animate_rotation_task(background, orig_img, orig_img2, 0, 100 * CV_PI, 0.1, pos_x, pos_y);
        break;
    }
    default:
        std::cerr << "Invalid selection. Use 1 for mood change animation, 2 for rotation animation, 3 for WatchTig animation." << std::endl;
        return -1;
    }

    return 0;
}
