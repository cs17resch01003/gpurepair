//pass
//--gridDim=[16,32,1] --blockDim=[8,4,2] --warp-sync=32

#include "common2.h"

__global__ void CUDAkernel2IDCT(float *dst, float *src, int ImgStride)
{
    __requires(ImgStride == 512);
    __shared__ float block[KER2_BLOCK_HEIGHT * KER2_SMEMBLOCK_STRIDE];

    int OffsThreadInRow = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    int OffsThreadInCol = threadIdx.z * BLOCK_SIZE;
    src += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) + blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
    dst += FMUL(blockIdx.y * KER2_BLOCK_HEIGHT + OffsThreadInCol, ImgStride) + blockIdx.x * KER2_BLOCK_WIDTH + OffsThreadInRow;
    float *bl_ptr = block + OffsThreadInCol * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow;

#pragma unroll

    for (unsigned int i = 0; i < BLOCK_SIZE; i++)
        bl_ptr[i * KER2_SMEMBLOCK_STRIDE] = src[i * ImgStride];

    //process rows
    CUDAsubroutineInplaceIDCTvector(block + (OffsThreadInCol + threadIdx.x) * KER2_SMEMBLOCK_STRIDE + OffsThreadInRow - threadIdx.x, 1);

    //process columns
    CUDAsubroutineInplaceIDCTvector(bl_ptr, KER2_SMEMBLOCK_STRIDE);

    for (unsigned int i = 0; i < BLOCK_SIZE; i++)
        dst[i * ImgStride] = bl_ptr[i * KER2_SMEMBLOCK_STRIDE];
}
