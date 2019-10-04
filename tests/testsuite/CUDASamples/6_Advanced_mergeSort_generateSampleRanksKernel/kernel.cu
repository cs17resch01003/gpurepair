//pass
//--gridDim=64               --blockDim=256

#include "common_merge.h"

template<uint sortDir> __global__ void generateSampleRanksKernel(uint *d_RanksA, uint *d_RanksB, uint *d_SrcKey, uint stride, uint N, uint threadCount);
template               __global__ void generateSampleRanksKernel<1>(uint *d_RanksA, uint *d_RanksB, uint *d_SrcKey, uint stride, uint N, uint threadCount);

template<uint sortDir> __global__ void generateSampleRanksKernel(
    uint *d_RanksA,
    uint *d_RanksB,
    uint *d_SrcKey,
    uint stride,
    uint N,
    uint threadCount
)
{
    __requires(stride & (stride - 1) == 0);
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const uint           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_SrcKey += segmentBase;
    d_RanksA += segmentBase / SAMPLE_STRIDE;
    d_RanksB += segmentBase / SAMPLE_STRIDE;

    const uint segmentElementsA = stride;
    const uint segmentElementsB = umin(stride, N - segmentBase - stride);
    const uint  segmentSamplesA = getSampleCount(segmentElementsA);
    const uint  segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        d_RanksA[i] = i * SAMPLE_STRIDE;
        d_RanksB[i] = binarySearchExclusive<sortDir>(
                          d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride,
                          segmentElementsB, nextPowerOfTwo(segmentElementsB)
                      );
    }

    if (i < segmentSamplesB)
    {
        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive<sortDir>(
                                                     d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0,
                                                     segmentElementsA, nextPowerOfTwo(segmentElementsA)
                                                 );
    }
}
