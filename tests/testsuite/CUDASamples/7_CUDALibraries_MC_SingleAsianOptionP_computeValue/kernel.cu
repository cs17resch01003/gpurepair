//pass
//--gridDim=195 --blockDim=128

#include "common.h"
__device__ float max(float, float);

template <typename Real> __device__ static __attribute__((always_inline)) Real reduce_sum(Real in);
template <typename Real> __global__ void computeValue(Real *const values, const Real *const paths, const AsianOption<Real> *const option, const unsigned int numSims, const unsigned int numTimesteps);
template                 __global__ void computeValue<float>(float *const values, const float *const paths, const AsianOption<float> *const option, const unsigned int numSims, const unsigned int numTimesteps);

template <typename Real>
__device__ static __attribute__((always_inline)) Real reduce_sum(Real in)
{
#if 0 // imperial edit
    SharedMemory<Real> sdata;
#else
    Real sdata[1];
#endif

    // Perform first level of reduction:
    // - Write to shared memory
    unsigned int ltid = threadIdx.x;

    sdata[ltid] = in;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2 ; s > 0 ; s >>= 1)
    {
        if (ltid < s)
        {
            sdata[ltid] += sdata[ltid + s];
        }

        __syncthreads();
    }

    return sdata[0];
}

template <typename Real>
__global__ void computeValue(Real *const values,
                             const Real *const paths,
                             const AsianOption<Real> *const option,
                             const unsigned int numSims,
                             const unsigned int numTimesteps)
{
    // Determine thread ID
    unsigned int bid = blockIdx.x;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    Real sumPayoffs = static_cast<Real>(0);

    for (unsigned int i = tid ; i < numSims ; i += step)
    {
        // Shift the input pointer
        const Real *path = paths + i;
        // Compute the arithmetic average
        Real avg = static_cast<Real>(0);

        for (unsigned int t = 0 ; t < numTimesteps ; t++, path += numSims)
        {
            avg += *path;
        }

        avg = avg * option->spot / numTimesteps;
        // Compute the payoff
        Real payoff = avg - option->strike;

        if (option->type == AsianOption<Real>::Put)
        {
            payoff = - payoff;
        }

        payoff = max(static_cast<Real>(0), payoff);
        // Accumulate payoff locally
        sumPayoffs += payoff;
    }

    // Reduce within the block
    sumPayoffs = reduce_sum<Real>(sumPayoffs);

    // Store the result
    if (threadIdx.x == 0)
    {
        values[bid] = sumPayoffs;
    }
}
