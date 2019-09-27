//xfail:NOT_ALL_VERIFIED
//--gridDim=64 --blockDim=256

template <class T> __global__ void reduce0(T *g_idata, T *g_odata, unsigned int n);
template __global__ void reduce0<int>(int *g_idata, int *g_odata, unsigned int n);

#include "common.h"

template <class T>
__global__ void
reduce0(T *g_idata, T *g_odata, unsigned int n)
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
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        // __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0){
     g_odata[blockIdx.x] = sdata[0];}
}
