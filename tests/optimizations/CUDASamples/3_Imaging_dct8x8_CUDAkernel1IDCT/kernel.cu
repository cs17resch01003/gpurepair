//pass
//--gridDim=[64,64,1]      --blockDim=[8,8,1]

#include "common1.h"

__global__ void CUDAkernel1IDCT(float *Dst, int ImgWidth, int OffsetXBlocks, int OffsetYBlocks)
{
    __requires(ImgWidth == 512);
    // Block index
    int bx = blockIdx.x + OffsetXBlocks;
    int by = blockIdx.y + OffsetYBlocks;

    // Thread index (current image pixel)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Texture coordinates
    const float tex_x = (float)((bx << BLOCK_SIZE_LOG2) + tx) + 0.5f;
    const float tex_y = (float)((by << BLOCK_SIZE_LOG2) + ty) + 0.5f;

    //copy current image pixel to the first block
    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = tex2D(TexSrc, tex_x, tex_y);

    //synchronize threads to make sure the block is copied
    __syncthreads();

    //calculate the multiplication of DCTv8matrix * A and place it in the second block
    float curelem = 0;
    int DCTv8matrixIndex = (ty << BLOCK_SIZE_LOG2) + 0;
    int CurBlockLocal1Index = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i=0; i<BLOCK_SIZE; i++)
    {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += 1;
        CurBlockLocal1Index += BLOCK_SIZE;
    }

    CurBlockLocal2[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;

    //synchronize threads to make sure the first 2 matrices are multiplied and the result is stored in the second block
    __syncthreads();

    //calculate the multiplication of (DCTv8matrix * A) * DCTv8matrixT and place it in the first block
    curelem = 0;
    int CurBlockLocal2Index = (ty << BLOCK_SIZE_LOG2) + 0;
    DCTv8matrixIndex = (tx << BLOCK_SIZE_LOG2) + 0;
#pragma unroll

    for (int i=0; i<BLOCK_SIZE; i++)
    {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += 1;
    }

    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;

    //synchronize threads to make sure the matrices are multiplied and the result is stored back in the first block
    __syncthreads();

    //copy current coefficient to its place in the result array
    Dst[ FMUL(((by << BLOCK_SIZE_LOG2) + ty), ImgWidth) + ((bx << BLOCK_SIZE_LOG2) + tx) ] = CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ];
}
