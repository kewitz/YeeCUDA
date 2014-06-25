/*
* The MIT License (MIT)
* Copyright (c) 2014 Leonardo Kewitz
*/
#ifndef CUDA_SNIPPETS
#define CUDA_SNIPPETS
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BSIZE 12

// Device function to return the index of a given thread within a block.
__device__ __inline__ unsigned int __tIndex(uint3 b, dim3 bd, uint3 t) {
	return t.x + t.y*bd.x + t.z*bd.x*bd.z;
};

// Return thread global index on a 2D space, given its global position and number of rows.
__device__ __inline__ unsigned int __tIndex2D(uint3 globalPosition, unsigned int rows){
	return globalPosition.y * rows + globalPosition.x;
};

// Calculate the thread global position, taken it's thread position within the block.
__device__ __inline__ uint3 __getGlobalPosition (uint3 blockIdx, dim3 blockDim, uint3 threadIdx) {
	return make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
};

void _assert (int assertion, char* message) {
    if (assertion == 0) {
        fprintf(stderr, "%s\n", message);
        exit(1);
    }
}

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stdout, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stdout, "cudaCheckError() failed at %s:%i : [%d] %s\n",
                 file, line, err, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stdout, "cudaCheckError() with sync failed at %s:%i : [%d] %s\n",
                 file, line, err, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}


#endif
