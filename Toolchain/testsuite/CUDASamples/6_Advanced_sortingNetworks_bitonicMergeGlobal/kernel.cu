//pass
//--gridDim=1024             --blockDim=512

#include "common.h"

__global__ void bitonicMergeGlobal(
    uint *d_DstKey,
    uint *d_DstVal,
    uint *d_SrcKey,
    uint *d_SrcVal,
    uint arrayLength,
    uint size,
    uint stride,
    uint dir
)
{
    __requires(arrayLength == 2048);
    __requires(stride == 1024);
    
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    //Bitonic merge
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    uint keyA = d_SrcKey[pos +      0];
    uint valA = d_SrcVal[pos +      0];
    uint keyB = d_SrcKey[pos + stride];
    uint valB = d_SrcVal[pos + stride];

    Comparator(
        keyA, valA,
        keyB, valB,
        ddd
    );

    d_DstKey[pos +      0] = keyA;
    d_DstVal[pos +      0] = valA;
    d_DstKey[pos + stride] = keyB;
    d_DstVal[pos + stride] = valB;
}
