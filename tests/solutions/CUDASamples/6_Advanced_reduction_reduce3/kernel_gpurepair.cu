//pass
//--gridDim=64 --blockDim=256

template <class T> __global__ void reduce3(T *g_idata, T *g_odata, unsigned int n);
template __global__ void reduce3<int>(int *g_idata, int *g_odata, unsigned int n);

#include "common.h"

template <class T>
__global__ void
reduce3(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n){
        mySum += g_idata[i+blockDim.x];
    }

    sdata[tid] = mySum;
    // __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
		__syncthreads();
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        // __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0){
     g_odata[blockIdx.x] = sdata[0];
    }
}
