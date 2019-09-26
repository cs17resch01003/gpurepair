//pass
//--gridDim=[52,1,1] --blockDim=32

#include "common.h"

__global__ void
generateTriangles(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned,
                  uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                  float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (i > activeVoxels - 1)
    {
        // can't return here because of syncthreads()
        i = activeVoxels - 1;
    }

#if SKIP_EMPTY_VOXELS
    uint voxel = compactedVoxelArray[i];
#else
    uint voxel = i;
#endif

    // compute position in 3d grid
    uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

    float3 p;
    p.x = -1.0f + (gridPos.x * voxelSize.x);
    p.y = -1.0f + (gridPos.y * voxelSize.y);
    p.z = -1.0f + (gridPos.z * voxelSize.z);

    // calculate cell vertex positions
    float3 v[8];
    v[0] = p;
    v[1] = p + make_float3(voxelSize.x, 0, 0);
    v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
    v[3] = p + make_float3(0, voxelSize.y, 0);
    v[4] = p + make_float3(0, 0, voxelSize.z);
    v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
    v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
    v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

    // evaluate field values
    float4 field[8];
    field[0] = fieldFunc4(v[0]);
    field[1] = fieldFunc4(v[1]);
    field[2] = fieldFunc4(v[2]);
    field[3] = fieldFunc4(v[3]);
    field[4] = fieldFunc4(v[4]);
    field[5] = fieldFunc4(v[5]);
    field[6] = fieldFunc4(v[6]);
    field[7] = fieldFunc4(v[7]);

    // recalculate flag
    // (this is faster than storing it in global memory)
    uint cubeindex;
    cubeindex =  uint(field[0].w < isoValue);
    cubeindex += uint(field[1].w < isoValue)*2;
    cubeindex += uint(field[2].w < isoValue)*4;
    cubeindex += uint(field[3].w < isoValue)*8;
    cubeindex += uint(field[4].w < isoValue)*16;
    cubeindex += uint(field[5].w < isoValue)*32;
    cubeindex += uint(field[6].w < isoValue)*64;
    cubeindex += uint(field[7].w < isoValue)*128;

    // find the vertices where the surface intersects the cube

#if USE_SHARED
    // use partioned shared memory to avoid using local memory
    __shared__ float3 vertlist[12*NTHREADS];
    __shared__ float3 normlist[12*NTHREADS];

    vertexInterp2(isoValue, v[0], v[1], field[0], field[1], vertlist[threadIdx.x], normlist[threadIdx.x]);
    vertexInterp2(isoValue, v[1], v[2], field[1], field[2], vertlist[threadIdx.x+NTHREADS], normlist[threadIdx.x+NTHREADS]);
    vertexInterp2(isoValue, v[2], v[3], field[2], field[3], vertlist[threadIdx.x+(NTHREADS*2)], normlist[threadIdx.x+(NTHREADS*2)]);
    vertexInterp2(isoValue, v[3], v[0], field[3], field[0], vertlist[threadIdx.x+(NTHREADS*3)], normlist[threadIdx.x+(NTHREADS*3)]);
    vertexInterp2(isoValue, v[4], v[5], field[4], field[5], vertlist[threadIdx.x+(NTHREADS*4)], normlist[threadIdx.x+(NTHREADS*4)]);
    vertexInterp2(isoValue, v[5], v[6], field[5], field[6], vertlist[threadIdx.x+(NTHREADS*5)], normlist[threadIdx.x+(NTHREADS*5)]);
    vertexInterp2(isoValue, v[6], v[7], field[6], field[7], vertlist[threadIdx.x+(NTHREADS*6)], normlist[threadIdx.x+(NTHREADS*6)]);
    vertexInterp2(isoValue, v[7], v[4], field[7], field[4], vertlist[threadIdx.x+(NTHREADS*7)], normlist[threadIdx.x+(NTHREADS*7)]);
    vertexInterp2(isoValue, v[0], v[4], field[0], field[4], vertlist[threadIdx.x+(NTHREADS*8)], normlist[threadIdx.x+(NTHREADS*8)]);
    vertexInterp2(isoValue, v[1], v[5], field[1], field[5], vertlist[threadIdx.x+(NTHREADS*9)], normlist[threadIdx.x+(NTHREADS*9)]);
    vertexInterp2(isoValue, v[2], v[6], field[2], field[6], vertlist[threadIdx.x+(NTHREADS*10)], normlist[threadIdx.x+(NTHREADS*10)]);
    vertexInterp2(isoValue, v[3], v[7], field[3], field[7], vertlist[threadIdx.x+(NTHREADS*11)], normlist[threadIdx.x+(NTHREADS*11)]);
    // __syncthreads();

#else
    float3 vertlist[12];
    float3 normlist[12];

    vertexInterp2(isoValue, v[0], v[1], field[0], field[1], vertlist[0], normlist[0]);
    vertexInterp2(isoValue, v[1], v[2], field[1], field[2], vertlist[1], normlist[1]);
    vertexInterp2(isoValue, v[2], v[3], field[2], field[3], vertlist[2], normlist[2]);
    vertexInterp2(isoValue, v[3], v[0], field[3], field[0], vertlist[3], normlist[3]);

    vertexInterp2(isoValue, v[4], v[5], field[4], field[5], vertlist[4], normlist[4]);
    vertexInterp2(isoValue, v[5], v[6], field[5], field[6], vertlist[5], normlist[5]);
    vertexInterp2(isoValue, v[6], v[7], field[6], field[7], vertlist[6], normlist[6]);
    vertexInterp2(isoValue, v[7], v[4], field[7], field[4], vertlist[7], normlist[7]);

    vertexInterp2(isoValue, v[0], v[4], field[0], field[4], vertlist[8], normlist[8]);
    vertexInterp2(isoValue, v[1], v[5], field[1], field[5], vertlist[9], normlist[9]);
    vertexInterp2(isoValue, v[2], v[6], field[2], field[6], vertlist[10], normlist[10]);
    vertexInterp2(isoValue, v[3], v[7], field[3], field[7], vertlist[11], normlist[11]);
#endif

    // output triangle vertices
    uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

    for (int i=0; i<numVerts; i++)
    {
        uint edge = tex1Dfetch(triTex, cubeindex*16 + i);

        uint index = numVertsScanned[voxel] + i;

        if (index < maxVerts)
        {
#if USE_SHARED
            pos[index] = make_float4(vertlist[(edge*NTHREADS)+threadIdx.x], 1.0f);
            norm[index] = make_float4(normlist[(edge*NTHREADS)+threadIdx.x], 0.0f);
#else
            pos[index] = make_float4(vertlist[edge], 1.0f);
            norm[index] = make_float4(normlist[edge], 0.0f);
#endif
        }
    }
}
