typedef unsigned int uint;
#define THREADBLOCK_SIZE 256

__device__ static __attribute__((always_inline)) uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size)
{
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1;
         __global_invariant(__write_implies(s_Data, __write_offset_bytes(s_Data)/sizeof(uint) == pos
                                                    | __write_offset_bytes(s_Data)/sizeof(uint) == pos - size)),
         offset < size; offset <<= 1)
    {
        // __syncthreads();
        uint t = s_Data[pos] + s_Data[pos - offset];
        // __syncthreads();
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

__device__ static __attribute__((always_inline)) uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size)
{
    return scan1Inclusive(idata, s_Data, size) - idata;
}
