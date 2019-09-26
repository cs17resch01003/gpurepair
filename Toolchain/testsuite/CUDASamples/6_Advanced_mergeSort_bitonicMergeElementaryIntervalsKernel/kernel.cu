//pass
//--gridDim=8 --blockDim=128

#define umin(x,y) (x < y ? x : y)
#define     SAMPLE_STRIDE 128
typedef unsigned int uint;
template<uint sortDir> __global__ void bitonicMergeElementaryIntervalsKernel(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint *d_LimitsA, uint *d_LimitsB, uint stride, uint N);
template               __global__ void bitonicMergeElementaryIntervalsKernel<1>(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint *d_LimitsA, uint *d_LimitsB, uint stride, uint N);

__device__ static __attribute__((always_inline)) uint iDivUp(uint a, uint b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

__device__ static __attribute__((always_inline)) uint getSampleCount(uint dividend)
{
    return iDivUp(dividend, SAMPLE_STRIDE);
}

template<uint sortDir> __device__ static __attribute__((always_inline)) void ComparatorExtended(
    uint &keyA,
    uint &valA,
    uint &flagA,
    uint &keyB,
    uint &valB,
    uint &flagB,
    uint arrowDir
)
{
    uint t;

    if (
        (!(flagA || flagB) && ((keyA > keyB) == arrowDir)) ||
        ((arrowDir == sortDir) && (flagA == 1)) ||
        ((arrowDir != sortDir) && (flagB == 1))
    )
    {
        t = keyA;
        keyA = keyB;
        keyB = t;
        t = valA;
        valA = valB;
        valB = t;
        t = flagA;
        flagA = flagB;
        flagB = t;
    }
}

template<uint sortDir> __global__ void bitonicMergeElementaryIntervalsKernel(
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
    __shared__ uint s_inf[2 * SAMPLE_STRIDE];

    const uint   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
    const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
    d_SrcKey += segmentBase;
    d_SrcVal += segmentBase;
    d_DstKey += segmentBase;
    d_DstVal += segmentBase;

    //Set up threadblock-wide parameters
    __shared__ uint startSrcA, lenSrcA, startSrcB, lenSrcB, startDst;

    if (threadIdx.x == 0)
    {
        uint segmentElementsA = stride;
        uint segmentElementsB = umin(stride, N - segmentBase - stride);
        uint  segmentSamplesA = stride / SAMPLE_STRIDE;
        uint  segmentSamplesB = getSampleCount(segmentElementsB);
        uint   segmentSamples = segmentSamplesA + segmentSamplesB;

        startSrcA    = d_LimitsA[blockIdx.x];
        startSrcB    = d_LimitsB[blockIdx.x];
        startDst     = startSrcA + startSrcB;

        uint endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
        uint endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
        lenSrcA      = endSrcA - startSrcA;
        lenSrcB      = endSrcB - startSrcB;
    }

    s_inf[threadIdx.x +             0] = 1;
    s_inf[threadIdx.x + SAMPLE_STRIDE] = 1;

    //Load input data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        s_key[threadIdx.x] = d_SrcKey[0 + startSrcA + threadIdx.x];
        s_val[threadIdx.x] = d_SrcVal[0 + startSrcA + threadIdx.x];
        s_inf[threadIdx.x] = 0;
    }

    //Prepare for bitonic merge by inversing the ordering
    if (threadIdx.x < lenSrcB)
    {
        s_key[2 * SAMPLE_STRIDE - 1 - threadIdx.x] = d_SrcKey[stride + startSrcB + threadIdx.x];
        s_val[2 * SAMPLE_STRIDE - 1 - threadIdx.x] = d_SrcVal[stride + startSrcB + threadIdx.x];
        s_inf[2 * SAMPLE_STRIDE - 1 - threadIdx.x] = 0;
    }

    //"Extended" bitonic merge
    for (uint stride = SAMPLE_STRIDE; stride > 0; stride >>= 1)
    {
        __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        ComparatorExtended<sortDir>(
            s_key[pos +      0], s_val[pos +      0], s_inf[pos +      0],
            s_key[pos + stride], s_val[pos + stride], s_inf[pos + stride],
            sortDir
        );
    }

    //Store sorted data
    __syncthreads();
    d_DstKey += startDst;
    d_DstVal += startDst;

    if (threadIdx.x < lenSrcA)
    {
        d_DstKey[threadIdx.x] = s_key[threadIdx.x];
        d_DstVal[threadIdx.x] = s_val[threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        d_DstKey[lenSrcA + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
        d_DstVal[lenSrcA + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
    }
}
