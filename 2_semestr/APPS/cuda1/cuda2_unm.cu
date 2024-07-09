/// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology usage with unified memory.
//
// Multiplication of elements in float array.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

// Demo kernel for array elements multiplication.
// Every thread selects one element and multiply it. 
__global__ void kernel_mult( float *t_array, float *t_array2, float *result,  int t_length, float t_mult )
{
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    // if grid is greater then length of array...
    if ( inx >= t_length ) return;

    result[inx] = t_array[ inx ] + t_array2[ inx ];
}

void cu_run_sum( float *t_array, float *t_array2, float *result,  int t_length, float t_mult )
{
    cudaError_t l_cerr;
    int l_threads = 128;
    int l_blocks = ( t_length + l_threads - 1 ) / l_threads;

    // Grid creation
    kernel_mult<<< l_blocks, l_threads >>>( t_array,t_array2, result, t_length, t_mult );
    
    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )

        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}


__global__ void kernel_prevod(char *buffer, long t_length) {
    int inx = blockDim.x * blockIdx.x + threadIdx.x;
    if (inx >= t_length) return;

    // Conversion table for lowercase letters to uppercase letters
    // const char conversion_table[26] = {
    //     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    //     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    // };

    const char conversionTable[256] = {
    // ASCII control characters
    '\0', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
    '\x08', '\t', '\n', '\x0B', '\x0C', '\r', '\x0E', '\x0F',
    '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17',
    '\x18', '\x19', '\x1A', '\x1B', '\x1C', '\x1D', '\x1E', '\x1F',
    ' ', '!', '"', '#', '$', '%', '&', '\'',
    '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', ':', ';', '<', '=', '>', '?',
    '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
    'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    '`', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
    'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z', '{', '|', '}', '~', '\x7F'
    };

    buffer[inx] = conversionTable[buffer[inx]];

    // If the character is a lowercase letter, convert it to uppercase
    // if (buffer[inx] >= 'a' && buffer[inx] <= 'z') {
    //     buffer[inx] = conversion_table[buffer[inx] - 'a'];
    // }     
}


void cu_run_prevod( char *buffer,  long t_length)
{
    cudaError_t l_cerr;
    int l_threads = 128;
    int l_blocks = ( t_length + l_threads - 1 ) / l_threads;

    // Grid creation
    kernel_prevod<<< l_blocks, l_threads >>>( buffer, t_length );
    
    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )

        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();

}



__global__ void kernel_matice(double *MaticeA, double *MaticeB, double *MaticeC, int t_length)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < t_length && col < t_length) {
        double sum = 0;
        for (int k = 0; k < t_length; ++k) {
            sum += MaticeA[row * t_length + k] * MaticeB[k * t_length + col];
        }
        MaticeC[row * t_length + col] = sum;
    }
}

void cu_run_matice(double *MaticeA, double *MaticeB, double *MaticeC, int t_length)
{
    cudaError_t l_cerr;
    
    
    int block_dim = 32;
    int grid_dim = (t_length + block_dim - 1) / block_dim;

    
    dim3 blocks(grid_dim, grid_dim);
    dim3 threads(block_dim, block_dim);
    
    kernel_matice<<< blocks, threads >>>(MaticeA, MaticeB, MaticeC, t_length);
    
    
    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
    
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}