//pass
//--gridDim=4370             --blockDim=64

#include "common.h"

//Data type used for input data fetches
typedef uint4 data_t;

//Count a byte into shared-memory storage
__device__ static __attribute__((always_inline)) void addByte(uchar *s_ThreadBase, uint data)
{
    s_ThreadBase[UMUL(data, HISTOGRAM64_THREADBLOCK_SIZE)]++;
}

//Count four bytes of a word
__device__ static __attribute__((always_inline)) void addWord(uchar *s_ThreadBase, uint data)
{
    //Only higher 6 bits of each byte matter, as this is a 64-bin histogram
    addByte(s_ThreadBase, (data >>  2) & 0x3FU);
    addByte(s_ThreadBase, (data >> 10) & 0x3FU);
    addByte(s_ThreadBase, (data >> 18) & 0x3FU);
    addByte(s_ThreadBase, (data >> 26) & 0x3FU);
}

__global__ void histogram64Kernel(uint *d_PartialHistograms, data_t *d_Data, uint dataCount)
{
    //Encode thread index in order to avoid bank conflicts in s_Hist[] access:
    //each group of SHARED_MEMORY_BANKS threads accesses consecutive shared memory banks
    //and the same bytes [0..3] within the banks
    //Because of this permutation block size should be a multiple of 4 * SHARED_MEMORY_BANKS
    const uint threadPos =
        ((threadIdx.x & ~(SHARED_MEMORY_BANKS * 4 - 1)) << 0) |
        ((threadIdx.x & (SHARED_MEMORY_BANKS     - 1)) << 2) |
        ((threadIdx.x & (SHARED_MEMORY_BANKS * 3)) >> 4);

    //Per-thread histogram storage
    __shared__ uchar s_Hist[HISTOGRAM64_THREADBLOCK_SIZE * HISTOGRAM64_BIN_COUNT];
    uchar *s_ThreadBase = s_Hist + threadPos;

    //Initialize shared memory (writing 32-bit words)
#pragma unroll

    for (uint i = 0; i < (HISTOGRAM64_BIN_COUNT / 4); i++)
    {
        ((uint *)s_Hist)[threadIdx.x + i * HISTOGRAM64_THREADBLOCK_SIZE] = 0;
    }

    //Read data from global memory and submit to the shared-memory histogram
    //Since histogram counters are byte-sized, every single thread can't do more than 255 submission
    // __syncthreads();

    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x);__global_invariant(__write_implies(s_Hist, (__write_offset_bytes(s_Hist)/sizeof(uchar) - threadPos)%HISTOGRAM64_THREADBLOCK_SIZE == 0)),__global_invariant(__read_implies(s_Hist, (__read_offset_bytes(s_Hist)/sizeof(uchar) - threadPos)%HISTOGRAM64_THREADBLOCK_SIZE == 0)),         pos < dataCount; pos += UMUL(blockDim.x, gridDim.x))
    {
        data_t data = d_Data[pos];
        addWord(s_ThreadBase, data.x);
        addWord(s_ThreadBase, data.y);
        addWord(s_ThreadBase, data.z);
        addWord(s_ThreadBase, data.w);
    }

    //Accumulate per-thread histograms into per-block and write to global memory
    // __syncthreads();

    if (threadIdx.x < HISTOGRAM64_BIN_COUNT)
    {
        uchar *s_HistBase = s_Hist + UMUL(threadIdx.x, HISTOGRAM64_THREADBLOCK_SIZE);

        uint sum = 0;
        uint pos = 4 * (threadIdx.x & (SHARED_MEMORY_BANKS - 1));

#pragma unroll

        for (uint i = 0; i < (HISTOGRAM64_THREADBLOCK_SIZE / 4); i++)
        {
            sum +=
                s_HistBase[pos + 0] +
                s_HistBase[pos + 1] +
                s_HistBase[pos + 2] +
                s_HistBase[pos + 3];
            pos = (pos + 4) & (HISTOGRAM64_THREADBLOCK_SIZE - 1);
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM64_BIN_COUNT + threadIdx.x] = sum;
    }
}
