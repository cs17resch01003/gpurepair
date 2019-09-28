//pass
//--gridDim=64 --blockDim=128 --warp-sync=32

#include "common.h"

template <unsigned int blockSize, bool nIsPow2> __global__ void reduceMultiPass(const float *g_idata, float *g_odata, unsigned int n);
template __global__ void reduceMultiPass<128, true>(const float *g_idata, float *g_odata, unsigned int n);


template <unsigned int blockSize, bool nIsPow2>
__global__ void
reduceMultiPass(const float *g_idata, float *g_odata, unsigned int n)
{
    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n);
}
