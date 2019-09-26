//pass
//--gridDim=[64,64,1]      --blockDim=[8,8,1]

#include "common_quantization.h"

__global__ void CUDAkernelQuantizationShort(short *SrcDst, int Stride)
{
     __requires(Stride == 512);
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index (current coefficient)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //copy current coefficient to the local variable
    short curCoef = SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx) ];
    short curQuant = Q[ty * BLOCK_SIZE + tx];

    //quantize the current coefficient
    if (curCoef < 0)
    {
        curCoef = -curCoef;
        curCoef += curQuant>>1;
        curCoef /= curQuant;
        curCoef = -curCoef;
    }
    else
    {
        curCoef += curQuant>>1;
        curCoef /= curQuant;
    }

    __syncthreads();

    curCoef = curCoef * curQuant;

    //copy quantized coefficient back to the DCT-plane
    SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx) ] = curCoef;
}
