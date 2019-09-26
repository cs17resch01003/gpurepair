//pass
//--gridDim=64 --blockDim=128

#include "common.h"

template <unsigned int blockSize, bool nIsPow2> __global__ void reduceSinglePass(const float *g_idata, float *g_odata, unsigned int n);
template                                        __global__ void reduceSinglePass<128, true>(const float *g_idata, float *g_odata, unsigned int n);

__device__ unsigned int retirementCount = 0;

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const float *g_idata, float *g_odata, unsigned int n)
{

    //
    // PHASE 1: Process all inputs assigned to this block
    //

    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n);

    //
    // PHASE 2: Last block finished will process all partial sums
    //

    if (gridDim.x > 1)
    {
        const unsigned int tid = threadIdx.x;
        __shared__ bool amLast;
        extern float __shared__ smem[];

        // wait until all outstanding memory instructions in this thread are finished
        __threadfence();

        // Thread 0 takes a ticket
        if (tid==0)
        {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            // If the ticket ID is equal to the number of blocks, we are the last block!
            amLast = (ticket == gridDim.x-1);
        }

        __syncthreads();

        // The last block sums the results of all other blocks
        if (amLast)
        {
            int i = tid;
            float mySum = 0;

            while (i < gridDim.x)
            {
                mySum += g_odata[i];
                i += blockSize;
            }

            reduceBlock<blockSize>(smem, mySum, tid);

            if (tid==0)
            {
                g_odata[0] = smem[0];

                // reset retirement count so that next run succeeds
                retirementCount = 0;
            }
        }
    }
}
