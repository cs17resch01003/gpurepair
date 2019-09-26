typedef unsigned int uint;
#define SHARED_SIZE_LIMIT 1024U
#define     SAMPLE_STRIDE 128
#define umin(x,y) (x < y ? x : y)

 __device__ static __attribute__((always_inline)) uint iDivUp(uint a, uint b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

 __device__ static __attribute__((always_inline)) uint getSampleCount(uint dividend)
{
    return iDivUp(dividend, SAMPLE_STRIDE);
}

#define W (sizeof(uint) * 8)
 __device__ static __attribute__((always_inline)) uint nextPowerOfTwo(uint x)
{
    /*
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    */
    return 1U << (W - __clz(x - 1));
}

template<uint sortDir>  __device__ static __attribute__((always_inline)) uint binarySearchInclusive(uint val, uint *data, uint L, uint stride)
{
    if (L == 0)
    {
        return 0;
    }

    uint pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
        {
            pos = newPos;
        }
    }

    return pos;
}

template<uint sortDir>  __device__ static __attribute__((always_inline)) uint binarySearchExclusive(uint val, uint *data, uint L, uint stride)
{
    if (L == 0)
    {
        return 0;
    }

    uint pos = 0;

    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);

        if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
        {
            pos = newPos;
        }
    }

    return pos;
}
