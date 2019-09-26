#define KNN_WINDOW_RADIUS   3
#define NLM_WINDOW_RADIUS   3
#define NLM_BLOCK_RADIUS    3
#define KNN_WINDOW_AREA     ( (2 * KNN_WINDOW_RADIUS + 1) * (2 * KNN_WINDOW_RADIUS + 1) )
#define NLM_WINDOW_AREA     ( (2 * NLM_WINDOW_RADIUS + 1) * (2 * NLM_WINDOW_RADIUS + 1) )
#define INV_KNN_WINDOW_AREA ( 1.0f / (float)KNN_WINDOW_AREA )
#define INV_NLM_WINDOW_AREA ( 1.0f / (float)NLM_WINDOW_AREA )

#define KNN_WEIGHT_THRESHOLD    0.02f
#define KNN_LERP_THRESHOLD      0.79f
#define NLM_WEIGHT_THRESHOLD    0.10f
#define NLM_LERP_THRESHOLD      0.10f

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

typedef unsigned int TColor;
__device__ float lerpf(float, float, float);

__device__ static __attribute__((always_inline)) TColor make_color(float r, float g, float b, float a);
__device__ static __attribute__((always_inline)) float vecLen(float4 a, float4 b);

texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;

__device__ static __attribute__((always_inline)) TColor make_color(float r, float g, float b, float a)
{
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}

__device__ static __attribute__((always_inline)) float vecLen(float4 a, float4 b)
{
    return (
               (b.x - a.x) * (b.x - a.x) +
               (b.y - a.y) * (b.y - a.y) +
               (b.z - a.z) * (b.z - a.z)
           );
}
