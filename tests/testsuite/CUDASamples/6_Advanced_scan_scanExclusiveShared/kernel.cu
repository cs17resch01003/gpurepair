//pass
//--gridDim=6624 --blockDim=256

#include "common.h"

__device__ static __attribute__((always_inline)) uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size)
{
    //Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    //Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}
__device__ static __attribute__((always_inline)) uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

__global__ void scanExclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint size
)
{
    __requires((size & (size - 1)) == 0);
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = d_Src[pos];

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, size);

    //Write back
    d_Dst[pos] = odata4;
}
