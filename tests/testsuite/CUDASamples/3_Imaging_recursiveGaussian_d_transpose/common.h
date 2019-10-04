typedef unsigned int uint;
__device__ float __saturatef(float);

__device__ static __attribute__((always_inline)) uint rgbaFloatToInt(float4 rgba);
__device__ static __attribute__((always_inline)) float4 rgbaIntToFloat(uint c);

#define CLAMP_TO_EDGE 1

// convert floating point rgba color to 32-bit integer
__device__ static __attribute__((always_inline)) uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

// convert from 32-bit int to float4
__device__ static __attribute__((always_inline)) float4 rgbaIntToFloat(uint c)
{
    float4 rgba;
    rgba.x = (c & 0xff) / 255.0f;
    rgba.y = ((c>>8) & 0xff) / 255.0f;
    rgba.z = ((c>>16) & 0xff) / 255.0f;
    rgba.w = ((c>>24) & 0xff) / 255.0f;
    return rgba;
}
