//pass
//--gridDim=64 --blockDim=256 2

template <class T> __global__ void reduce2(T *g_idata, T *g_odata, unsigned int n);
template __global__ void reduce2<int>(int *g_idata, int *g_odata, unsigned int n);

#include "common.h"

template <class T>
__global__ void
reduce2(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

if(i<n){
        sdata[tid]=g_idata[i];
    }
    else{
     sdata[tid]= 0;
    }
    // __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        // __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0){
     g_odata[blockIdx.x] = sdata[0];
    }
}
