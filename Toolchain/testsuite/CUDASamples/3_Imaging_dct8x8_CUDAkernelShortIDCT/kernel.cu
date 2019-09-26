//pass
//--gridDim=[16,16,1]      --blockDim=[8,4,4]

#include "common_short.h"

__global__ void CUDAkernelShortIDCT(short *SrcDst, int ImgStride)
{
    __requires(ImgStride == 512);

    __shared__ short block[KERS_BLOCK_HEIGHT * KERS_SMEMBLOCK_STRIDE];
    int    OffsThreadInRow = IMAD(threadIdx.y, BLOCK_SIZE, threadIdx.x);
    int    OffsThreadInCol = IMUL(threadIdx.z, BLOCK_SIZE);
    int OffsThrRowPermuted = (OffsThreadInRow & 0xFFFFFFE0) | ((OffsThreadInRow << 1) | (OffsThreadInRow >> 4) & 0x1) & 0x1F;

    SrcDst += IMAD(IMAD(blockIdx.y, KERS_BLOCK_HEIGHT, OffsThreadInCol), ImgStride, IMAD(blockIdx.x, KERS_BLOCK_WIDTH, OffsThreadInRow * 2));
    short *bl_ptr = block + IMAD(OffsThreadInCol, KERS_SMEMBLOCK_STRIDE, OffsThreadInRow * 2);

    //load data to shared memory (only first half of threads in each row performs data moving (each thread moves 2 shorts)
    if (OffsThreadInRow < KERS_BLOCK_WIDTH_HALF)
    {
#pragma unroll

        for (int i = 0;
             #define SrcDstOffset (IMAD(IMAD(blockIdx.y, KERS_BLOCK_HEIGHT, OffsThreadInCol), ImgStride, IMAD(blockIdx.x, KERS_BLOCK_WIDTH, OffsThreadInRow * 2)) \
                                                                                                                                                         * sizeof(short))
             #define blockOffset (IMAD(OffsThreadInCol, KERS_SMEMBLOCK_STRIDE, OffsThreadInRow * 2) * sizeof(short))
             __global_invariant(__write_implies(block, (__write_offset_bytes(block) - blockOffset)/sizeof(int)%(KERS_SMEMBLOCK_STRIDE/2) == 0)),
             __global_invariant(__write_implies(block, (__write_offset_bytes(block) - blockOffset)/sizeof(int)/(KERS_SMEMBLOCK_STRIDE/2) < BLOCK_SIZE)),
             __global_invariant(__read_implies(SrcDst, (__read_offset_bytes(SrcDst) - SrcDstOffset)/sizeof(int)%(ImgStride/2) == 0)),
             __global_invariant(__read_implies(SrcDst, (__read_offset_bytes(SrcDst) - SrcDstOffset)/sizeof(int)/(ImgStride/2) < BLOCK_SIZE)),
             i < BLOCK_SIZE; i++)
            ((int *)bl_ptr)[i * (KERS_SMEMBLOCK_STRIDE / 2)] = ((int *)SrcDst)[i * (ImgStride / 2)];
    }

    __syncthreads();
    CUDAshortInplaceIDCT(block + OffsThreadInCol * KERS_SMEMBLOCK_STRIDE + OffsThrRowPermuted, KERS_SMEMBLOCK_STRIDE);
    __syncthreads();
    CUDAshortInplaceIDCT((unsigned int *)(block + OffsThreadInRow * KERS_SMEMBLOCK_STRIDE + OffsThreadInCol));
    __syncthreads();

    //store data to global memory (only first half of threads in each row performs data moving (each thread moves 2 shorts)
    if (OffsThreadInRow < KERS_BLOCK_WIDTH_HALF)
    {
#pragma unroll

        for (int i = 0;
             __global_invariant(__write_implies(SrcDst, (__write_offset_bytes(SrcDst) - SrcDstOffset)/sizeof(int)%(ImgStride/2) == 0)),
             __global_invariant(__write_implies(SrcDst, (__write_offset_bytes(SrcDst) - SrcDstOffset)/sizeof(int)/(ImgStride/2) < BLOCK_SIZE)),
             i < BLOCK_SIZE; i++)
            ((int *)SrcDst)[i * (ImgStride / 2)] = ((int *)bl_ptr)[i * (KERS_SMEMBLOCK_STRIDE / 2)];
    }
}
