typedef unsigned char uchar;
typedef unsigned int uint;

__device__ static __attribute__((always_inline)) float tangle(float x, float y, float z);
__device__ static __attribute__((always_inline)) float fieldFunc(float3 p);
__device__ static __attribute__((always_inline)) float4 fieldFunc4(float3 p);
__device__ static __attribute__((always_inline)) float sampleVolume(uchar *data, uint3 p, uint3 gridSize);
__device__ static __attribute__((always_inline)) uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask);
__device__ static __attribute__((always_inline)) float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1);
__device__ static __attribute__((always_inline)) void vertexInterp2(float isolevel, float3 p0, float3 p1, float4 f0, float4 f1, float3 &p, float3 &n);
__device__ static __attribute__((always_inline)) float3 calcNormal(float3 *v0, float3 *v1, float3 *v2);

texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;
texture<uchar, 1, cudaReadModeNormalizedFloat> volumeTex;

#if 1
__device__ static __attribute__((always_inline))
float tangle(float x, float y, float z)
{
    x *= 3.0f;
    y *= 3.0f;
    z *= 3.0f;
    return (x*x*x*x - 5.0f*x*x +y*y*y*y - 5.0f*y*y +z*z*z*z - 5.0f*z*z + 11.8f) * 0.2f + 0.5f;
}

__device__ static __attribute__((always_inline))
float fieldFunc(float3 p)
{
    return tangle(p.x, p.y, p.z);
}

__device__ static __attribute__((always_inline))
float4 fieldFunc4(float3 p)
{
    float v = tangle(p.x, p.y, p.z);
    const float d = 0.001f;
    float dx = tangle(p.x + d, p.y, p.z) - v;
    float dy = tangle(p.x, p.y + d, p.z) - v;
    float dz = tangle(p.x, p.y, p.z + d) - v;
    return make_float4(dx, dy, dz, v);
}

__device__ static __attribute__((always_inline))
float sampleVolume(uchar *data, uint3 p, uint3 gridSize)
{
    p.x = min(p.x, gridSize.x - 1);
    p.y = min(p.y, gridSize.y - 1);
    p.z = min(p.z, gridSize.z - 1);
    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
    //    return (float) data[i] / 255.0f;
    return tex1Dfetch(volumeTex, i);
}

__device__ static __attribute__((always_inline))
uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
{
    uint3 gridPos;
    gridPos.x = i & gridSizeMask.x;
    gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
    gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
    return gridPos;
}

__device__ static __attribute__((always_inline))
float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
    float t = (isolevel - f0) / (f1 - f0);
    return lerp(p0, p1, t);
}

__device__ static __attribute__((always_inline))
void vertexInterp2(float isolevel, float3 p0, float3 p1, float4 f0, float4 f1, float3 &p, float3 &n)
{
    float t = (isolevel - f0.w) / (f1.w - f0.w);
    p = lerp(p0, p1, t);
    n.x = lerp(f0.x, f1.x, t);
    n.y = lerp(f0.y, f1.y, t);
    n.z = lerp(f0.z, f1.z, t);
    //    n = normalize(n);
}

__device__ static __attribute__((always_inline))
float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
    float3 edge0 = *v1 - *v0;
    float3 edge1 = *v2 - *v0;
    // note - it's faster to perform normalization in vertex shader rather than here
    return cross(edge0, edge1);
}
#endif
