//pass
//--gridDim=[256,1,1] --blockDim=[128,1,1]

#include "common.h"

__global__ void
classifyVoxel(uint *voxelVerts, uint *voxelOccupied, uchar *volume,
              uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
              float3 voxelSize, float isoValue)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

    // read field values at neighbouring grid vertices
#if SAMPLE_VOLUME
    float field[8];
    field[0] = sampleVolume(volume, gridPos, gridSize);
    field[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
    field[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
    field[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
    field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
    field[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
    field[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
    field[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);
#else
    float3 p;
    p.x = -1.0f + (gridPos.x * voxelSize.x);
    p.y = -1.0f + (gridPos.y * voxelSize.y);
    p.z = -1.0f + (gridPos.z * voxelSize.z);

    float field[8];
    field[0] = fieldFunc(p);
    field[1] = fieldFunc(p + make_float3(voxelSize.x, 0, 0));
    field[2] = fieldFunc(p + make_float3(voxelSize.x, voxelSize.y, 0));
    field[3] = fieldFunc(p + make_float3(0, voxelSize.y, 0));
    field[4] = fieldFunc(p + make_float3(0, 0, voxelSize.z));
    field[5] = fieldFunc(p + make_float3(voxelSize.x, 0, voxelSize.z));
    field[6] = fieldFunc(p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z));
    field[7] = fieldFunc(p + make_float3(0, voxelSize.y, voxelSize.z));
#endif

    // calculate flag indicating if each vertex is inside or outside isosurface
    uint cubeindex;
    cubeindex =  uint(field[0] < isoValue);
    cubeindex += uint(field[1] < isoValue)*2;
    cubeindex += uint(field[2] < isoValue)*4;
    cubeindex += uint(field[3] < isoValue)*8;
    cubeindex += uint(field[4] < isoValue)*16;
    cubeindex += uint(field[5] < isoValue)*32;
    cubeindex += uint(field[6] < isoValue)*64;
    cubeindex += uint(field[7] < isoValue)*128;

    // read number of vertices from texture
    uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

    if (i < numVoxels)
    {
        voxelVerts[i] = numVerts;
        voxelOccupied[i] = (numVerts > 0);
    }
}
