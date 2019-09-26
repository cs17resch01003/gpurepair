//pass
//--gridDim=[52,1,1] --blockDim=32

#include "common.h"

__global__ void
generateTriangles2(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, uchar *volume,
                   uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
                   float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (i > activeVoxels - 1)
    {
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
    // evaluate field values
    float field[8];
    field[0] = fieldFunc(v[0]);
    field[1] = fieldFunc(v[1]);
    field[2] = fieldFunc(v[2]);
    field[3] = fieldFunc(v[3]);
    field[4] = fieldFunc(v[4]);
    field[5] = fieldFunc(v[5]);
    field[6] = fieldFunc(v[6]);
    field[7] = fieldFunc(v[7]);
#endif

    // recalculate flag
    uint cubeindex;
    cubeindex =  uint(field[0] < isoValue);
    cubeindex += uint(field[1] < isoValue)*2;
    cubeindex += uint(field[2] < isoValue)*4;
    cubeindex += uint(field[3] < isoValue)*8;
    cubeindex += uint(field[4] < isoValue)*16;
    cubeindex += uint(field[5] < isoValue)*32;
    cubeindex += uint(field[6] < isoValue)*64;
    cubeindex += uint(field[7] < isoValue)*128;

    // find the vertices where the surface intersects the cube

#if USE_SHARED
    // use shared memory to avoid using local
    __shared__ float3 vertlist[12*NTHREADS];

    vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
    vertlist[NTHREADS+threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
    vertlist[(NTHREADS*2)+threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
    vertlist[(NTHREADS*3)+threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
    vertlist[(NTHREADS*4)+threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
    vertlist[(NTHREADS*5)+threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
    vertlist[(NTHREADS*6)+threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
    vertlist[(NTHREADS*7)+threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
    vertlist[(NTHREADS*8)+threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
    vertlist[(NTHREADS*9)+threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
    vertlist[(NTHREADS*10)+threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
    vertlist[(NTHREADS*11)+threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
    // __syncthreads();
#else

    float3 vertlist[12];

    vertlist[0] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
    vertlist[1] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
    vertlist[2] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
    vertlist[3] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);

    vertlist[4] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
    vertlist[5] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
    vertlist[6] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
    vertlist[7] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);

    vertlist[8] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
    vertlist[9] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
    vertlist[10] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
    vertlist[11] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
#endif

    // output triangle vertices
    uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

    for (int i=0; i<numVerts; i+=3)
    {
        uint index = numVertsScanned[voxel] + i;

        // imperial edit: replaced array of three float3 pointers with three distinct float3 pointers
        
        float3* v0;
        float3* v1;
        float3* v2;

        uint edge;
        edge = tex1Dfetch(triTex, (cubeindex*16) + i);
#if USE_SHARED
        v0 = &vertlist[(edge*NTHREADS)+threadIdx.x];
#else
        v0 = &vertlist[edge];
#endif

        edge = tex1Dfetch(triTex, (cubeindex*16) + i + 1);
#if USE_SHARED
        v1 = &vertlist[(edge*NTHREADS)+threadIdx.x];
#else
        v1 = &vertlist[edge];
#endif

        edge = tex1Dfetch(triTex, (cubeindex*16) + i + 2);
#if USE_SHARED
        v2 = &vertlist[(edge*NTHREADS)+threadIdx.x];
#else
        v2 = &vertlist[edge];
#endif

        // calculate triangle surface normal
        float3 n = calcNormal(v0, v1, v2);

        if (index < (maxVerts - 3))
        {
#if 0
            pos[index] = make_float4(*v[0], 1.0f);
            norm[index] = make_float4(n, 0.0f);

            pos[index+1] = make_float4(*v[1], 1.0f);
            norm[index+1] = make_float4(n, 0.0f);

            pos[index+2] = make_float4(*v[2], 1.0f);
            norm[index+2] = make_float4(n, 0.0f);
#else
            float4 v1, v2, v3, v4, v5, v6;
            pos[index] = v1;
            norm[index] = v2;

            pos[index+1] = v3;
            norm[index+1] = v4;

            pos[index+2] = v5;
            norm[index+2] = v6;
#endif
        }
    }
}
