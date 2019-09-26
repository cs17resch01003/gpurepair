//pass
//--gridDim=8 --blockDim=512

#define SHARED_SIZE_LIMIT 1024U
#define     SAMPLE_STRIDE 128
typedef unsigned int uint;

__device__ static __attribute__((always_inline)) void Comparator(
    uint &keyA,
    uint &valA,
    uint &keyB,
    uint &valB,
    uint arrowDir
)
{
    uint t;

    if ((keyA > keyB) == arrowDir)
    {
        t = keyA;
        keyA = keyB;
        keyB = t;
        t = valA;
        valA = valB;
        valB = t;
    }
}

__global__ void bitonicSortSharedKernel(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength,
    uint sortDir
)
{
    __requires(arrayLength & (arrayLength - 1) == 0);
    //Shared memory storage for one or more short vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    //Offset to the beginning of subbatch and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
    s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size < arrayLength; size <<= 1)
    {
        //Bitonic merge
        uint dir = (threadIdx.x & (size / 2)) != 0;

        for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                dir
            );
        }
    }

    //ddd == sortDir for the last bitonic merge step
    {
        for (uint stride = arrayLength / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_key[pos +      0], s_val[pos +      0],
                s_key[pos + stride], s_val[pos + stride],
                sortDir
            );
        }
    }

    __syncthreads();
    d_DstKey[                      0] = s_key[threadIdx.x +                       0];
    d_DstVal[                      0] = s_val[threadIdx.x +                       0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}
