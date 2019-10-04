//pass
//--gridDim=[64,64] --blockDim=[16,16]

#include "common.h"

__global__ void copy(float *odata, float *idata, int width, int height, int nreps)
{
    __requires(width == 1024);
    __requires(height == 1024);
    __requires(nreps == 1);

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index  = xIndex + width*yIndex;

    for (int r=0; r < nreps; r++)
    {
        for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
        {
            odata[index+i*width] = idata[index+i*width];
        }
    }
}
