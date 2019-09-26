//pass
//--gridDim=64 --blockDim=256 2

template <class T> __global__ void reduce1(T *g_idata, T *g_odata, unsigned int n);
template __global__ void reduce1<int>(int *g_idata, int *g_odata, unsigned int n);

#include "common.h"

template <class T>
__global__ void
reduce1(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    // __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }

        // __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0){
     g_odata[blockIdx.x] = sdata[0];
    }
}
