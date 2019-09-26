//pass
//--gridDim=240              --blockDim=192

#include "common.h"

#define USE_SMEM_ATOMICS 1

#if(!USE_SMEM_ATOMICS)
#define TAG_MASK ( (1U << (UINT_BITS - LOG2_WARP_SIZE)) - 1U )

__device__ static __attribute__((always_inline)) void addByte(volatile uint *s_WarpHist, uint data, uint threadTag)
{
    uint count;

    do
    {
        count = s_WarpHist[data] & TAG_MASK;
        count = threadTag | (count + 1);
        s_WarpHist[data] = count;
    }
    while (s_WarpHist[data] != count);
}
#else

#define TAG_MASK 0xFFFFFFFFU
__device__ static __attribute__((always_inline)) void addByte(uint *s_WarpHist, uint data, uint threadTag)
{
    atomicAdd(s_WarpHist + data, 1);
}
#endif

__device__ static __attribute__((always_inline)) void addWord(uint *s_WarpHist, uint data, uint tag)
{
    addByte(s_WarpHist, (data >>  0) & 0xFFU, tag);
    addByte(s_WarpHist, (data >>  8) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

__global__ void histogram256Kernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount)
{
    //Per-warp subhistogram storage
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    //Clear shared memory storage for current threadblock before processing
#pragma unroll

    for (uint i = 0;__invariant(__write_implies(s_Hist, ((__write_offset_bytes(s_Hist)/sizeof(uint))%HISTOGRAM256_THREADBLOCK_SIZE) == threadIdx.x)),i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
    {
        s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
    }

    //Cycle through the entire data set, update subhistograms for each warp
    const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

    // __syncthreads();

    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x))
    {
        uint data = d_Data[pos];
        addWord(s_WarpHist, data, tag);
    }

    //Merge per-warp histograms into per-block and write to global memory
    // __syncthreads();

    for (uint bin = threadIdx.x; __global_invariant(bin % blockDim.x == threadIdx.x),__global_invariant(__write_implies(d_PartialHistograms, __write_offset_bytes(d_PartialHistograms)/sizeof(uint) % HISTOGRAM256_BIN_COUNT % blockDim.x == threadIdx.x)),bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE)
    {
        uint sum = 0;

        for (uint i = 0; i < WARP_COUNT; i++)
        {
            sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}
