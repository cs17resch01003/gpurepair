//pass
//--gridDim=195 --blockDim=128

__global__ void initRNG(curandState *const rngStates,
                        const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}
