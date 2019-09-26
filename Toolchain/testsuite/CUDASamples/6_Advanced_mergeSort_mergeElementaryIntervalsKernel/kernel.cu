//pass
//--gridDim=32768            --blockDim=128

#include "common_merge.h"
template<uint sortDir> __global__ void mergeElementaryIntervalsKernel(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint *d_LimitsA, uint *d_LimitsB, uint stride, uint N);
template               __global__ void mergeElementaryIntervalsKernel<1>(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint *d_LimitsA, uint *d_LimitsB, uint stride, uint N);

template<uint sortDir> __device__ static __attribute__((always_inline)) void merge(
    uint *dstKey,
    uint *dstVal,
    uint *srcAKey,
    uint *srcAVal,
    uint *srcBKey,
    uint *srcBVal,
    uint lenA,
    uint nPowTwoLenA,
    uint lenB,
    uint nPowTwoLenB
)
{
    uint keyA, valA, keyB, valB, dstPosA, dstPosB;

    if (threadIdx.x < lenA)
    {
        keyA = srcAKey[threadIdx.x];
        valA = srcAVal[threadIdx.x];
        dstPosA = binarySearchExclusive<sortDir>(keyA, srcBKey, lenB, nPowTwoLenB) + threadIdx.x;
    }

    if (threadIdx.x < lenB)
    {
        keyB = srcBKey[threadIdx.x];
        valB = srcBVal[threadIdx.x];
        dstPosB = binarySearchInclusive<sortDir>(keyB, srcAKey, lenA, nPowTwoLenA) + threadIdx.x;
    }

    __syncthreads();

    if (threadIdx.x < lenA)
    {
        dstKey[dstPosA] = keyA;
        dstVal[dstPosA] = valA;
    }

    if (threadIdx.x < lenB)
    {
        dstKey[dstPosB] = keyB;
        dstVal[dstPosB] = valB;
    }
}


template<uint sortDir> __global__ void mergeElementaryIntervalsKernel(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint stride,
    uint N
)
{
    __requires((stride & (stride - 1)) == 0);
    __requires(stride < N);
    __requires(stride > 1);
    __shared__ uint s_key[2 * SAMPLE_STRIDE];
    __shared__ uint s_val[2 * SAMPLE_STRIDE];

    const uint   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
    const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
    d_SrcKey += segmentBase;
    d_SrcVal += segmentBase;
    d_DstKey += segmentBase;
    d_DstVal += segmentBase;

    //Set up threadblock-wide parameters
    __shared__ uint startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

    if (threadIdx.x == 0)
    {
        uint segmentElementsA = stride;
        uint segmentElementsB = umin(stride, N - segmentBase - stride);
        uint  segmentSamplesA = getSampleCount(segmentElementsA);
        uint  segmentSamplesB = getSampleCount(segmentElementsB);
        uint   segmentSamples = segmentSamplesA + segmentSamplesB;

        startSrcA    = d_LimitsA[blockIdx.x];
        startSrcB    = d_LimitsB[blockIdx.x];
        uint endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
        uint endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
        lenSrcA      = endSrcA - startSrcA;
        lenSrcB      = endSrcB - startSrcB;
        startDstA    = startSrcA + startSrcB;
        startDstB    = startDstA + lenSrcA;
    }

    //Load main input data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        s_key[threadIdx.x +             0] = d_SrcKey[0 + startSrcA + threadIdx.x];
        s_val[threadIdx.x +             0] = d_SrcVal[0 + startSrcA + threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x];
        s_val[threadIdx.x + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadIdx.x];
    }

    //Merge data in shared memory
    __syncthreads();
    merge<sortDir>(
        s_key,
        s_val,
        s_key + 0,
        s_val + 0,
        s_key + SAMPLE_STRIDE,
        s_val + SAMPLE_STRIDE,
        lenSrcA, SAMPLE_STRIDE,
        lenSrcB, SAMPLE_STRIDE
    );

    //Store merged data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x];
        d_DstVal[startDstA + threadIdx.x] = s_val[threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
        d_DstVal[startDstB + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
    }
}
