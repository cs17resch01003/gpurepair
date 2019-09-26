//pass
//--gridDim=26               --blockDim=256

#include "common.h"

__global__ void scanExclusiveShared2(
    uint *d_Buf,
    uint *d_Dst,
    uint *d_Src,
    uint N,
    uint arrayLength
)
{
    __requires(N == 6656);
    __requires((arrayLength & (arrayLength - 1)) == 0);
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    //Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    uint idata = 0;

    if (pos < N)
        idata =
            d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] +
            d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

    //Compute
    uint odata = scan1Exclusive(idata, s_Data, arrayLength);

    //Avoid out-of-bound access
    if (pos < N)
    {
        d_Buf[pos] = odata;
    }
}
