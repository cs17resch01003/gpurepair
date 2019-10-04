template <unsigned int blockSize>
__device__ static __attribute__((always_inline)) void
reduceBlock(volatile float *sdata, float mySum, const unsigned int tid)
{
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        if (blockSize >=  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 32];
        }

        if (blockSize >=  32)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 16];
        }

        if (blockSize >=  16)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  8];
        }

        if (blockSize >=   8)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  4];
        }

        if (blockSize >=   4)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  2];
        }

        if (blockSize >=   2)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  1];
        }
    }
}

template <unsigned int blockSize, bool nIsPow2>
__device__ static __attribute__((always_inline)) void
reduceBlocks(const float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // do reduction in shared mem
    reduceBlock<blockSize>(sdata, mySum, tid);

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
