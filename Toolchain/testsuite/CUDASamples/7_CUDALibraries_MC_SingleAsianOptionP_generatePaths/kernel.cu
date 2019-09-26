//pass
//--gridDim=195 --blockDim=128

#include "common.h"

template <typename Real> __global__ void generatePaths(Real *const paths, curandState *const rngStates, const AsianOption<Real> *const option, const unsigned int numSims, const unsigned int numTimesteps);
template                 __global__ void generatePaths<float>(float *const paths, curandState *const rngStates, const AsianOption<float> *const option, const unsigned int numSims, const unsigned int numTimesteps);

__device__ static __attribute__((always_inline)) float getPathStep(float &drift, float &diffusion, curandState &state)
{
    return expf(drift + diffusion * curand_normal(&state));
}
__device__ static __attribute__((always_inline)) double getPathStep(double &drift, double &diffusion, curandState &state)
{
    return exp(drift + diffusion * curand_normal_double(&state));
}

// Path generation kernel
template <typename Real>
__global__ void generatePaths(Real *const paths,
                              curandState *const rngStates,
                              const AsianOption<Real> *const option,
                              const unsigned int numSims,
                              const unsigned int numTimesteps)
{
    __requires(numSims == 100000);
    __requires(numTimesteps == 87);

    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // Compute parameters
    Real drift     = (option->r - static_cast<Real>(0.5) * option->sigma * option->sigma) * option->dt;
    Real diffusion = option->sigma * sqrt(option->dt);

    // Initialise the RNG
    curandState localState = rngStates[tid];

    for (unsigned int i = tid ;
         __global_invariant(i%step == tid),
         __global_invariant(__write_implies(paths, (__write_offset_bytes(paths)/sizeof(Real))%numSims%step == tid)),
         i < numSims ; i += step)
    {
        // Shift the output pointer
        Real *output = paths + i;

        // Simulate the path
        Real s = static_cast<Real>(1);

        for (unsigned int t = 0 ;
             __invariant(__ptr_offset_bytes(output)/sizeof(Real) - i == t * numSims),
             __global_invariant(__write_implies(paths, (__write_offset_bytes(paths)/sizeof(Real))%numSims%step == tid)),
             t < numTimesteps ; t++, output += numSims)
        {
            s *= getPathStep(drift, diffusion, localState);
            *output = s;
        }
    }
}
