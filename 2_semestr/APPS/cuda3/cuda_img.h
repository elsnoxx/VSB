#pragma once

#include <opencv2/core/mat.hpp>

// Structure definition for exchanging data between Host and Device
struct CudaImg {
    uint3 m_size;             // size of picture (width, height, depth)
    void *m_p_void;           // data of picture
    int channels;             // number of color channels (1, 3, 4)

    union {
        uchar1 *m_p_uchar1;   // data of picture for 1 channel
        uchar3 *m_p_uchar3;   // data of picture for 3 channels
        uchar4 *m_p_uchar4;   // data of picture for 4 channels
    };
};
