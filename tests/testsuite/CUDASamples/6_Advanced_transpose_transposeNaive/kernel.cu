//pass
//--gridDim=[64,64] --blockDim=[16,16]

#include "common.h"

__global__ void transposeNaive(float *odata, float *idata, int width, int height, int nreps)
{
    __requires(height == 1024);

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in  = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;

    for (int r=0; r < nreps; r++)
    {
        for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
        {
            odata[index_out+i] = idata[index_in+i*width];
        }
    }
}
