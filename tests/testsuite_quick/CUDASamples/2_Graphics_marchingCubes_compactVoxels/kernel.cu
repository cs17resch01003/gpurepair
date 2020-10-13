//pass
//--gridDim=[256,1,1] --blockDim=[128,1,1]

#include "common.h"

#define tid ((((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x) + threadIdx.x)
#define other_tid __other_int(tid)

__global__ void
compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
    // monotonic prefix sum specification
    __requires(__implies(tid < other_tid,              voxelOccupiedScan[tid] + voxelOccupied[tid] <= voxelOccupiedScan[other_tid]));
    __requires(__implies(tid < other_tid, __add_noovfl(voxelOccupiedScan[tid], voxelOccupied[tid])));

    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (voxelOccupied[i] && (i < numVoxels))
    {
        compactedVoxelArray[ voxelOccupiedScan[i] ] = i;
    }
}
