//pass
//--gridDim=[64,64] --blockDim=[16,16]

#include "common.h"

__global__ void transposeCoarseGrained(float *odata, float *idata, int width, int height, int nreps)
{
    __requires(width == 1024);
    __requires(height == 1024);
    __requires(nreps == 1);

    __shared__ float block[TILE_DIM][TILE_DIM+1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int r=0; r<nreps; r++)
    {
        for (int i=0; i<TILE_DIM; i += BLOCK_ROWS)
        {
            block[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
        }

        __syncthreads();

        for (int i=0; i<TILE_DIM; i += BLOCK_ROWS)
        {
            odata[index_out+i*height] = block[threadIdx.y+i][threadIdx.x];
        }

        // IMPERIAL EDIT: add barrier
        __syncthreads();
    }
}
