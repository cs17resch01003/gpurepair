//pass
//--gridDim=195 --blockDim=128

template <typename Real> __global__ void computeValue(unsigned int *const results, curandState *const rngStates, const unsigned int numSims);
template                 __global__ void computeValue<float>(unsigned int *const results, curandState *const rngStates, const unsigned int numSims);

__device__ static __attribute__((always_inline)) unsigned int reduce_sum(unsigned int in);
__device__ static __attribute__((always_inline)) void getPoint(float &x, float &y, curandState &state);
__device__ static __attribute__((always_inline)) void getPoint(double &x, double &y, curandState &state);

__device__ static __attribute__((always_inline)) unsigned int reduce_sum(unsigned int in)
{
    extern __shared__ unsigned int sdata[];

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

__device__ static __attribute__((always_inline)) void getPoint(float &x, float &y, curandState &state)
{
    x = curand_uniform(&state);
    y = curand_uniform(&state);
}
__device__ static __attribute__((always_inline)) void getPoint(double &x, double &y, curandState &state)
{
    x = curand_uniform_double(&state);
    y = curand_uniform_double(&state);
}

// Estimator kernel
template <typename Real>
__global__ void computeValue(unsigned int *const results,
                             curandState *const rngStates,
                             const unsigned int numSims)
{
    // Determine thread ID
    unsigned int bid = blockIdx.x;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // Initialise the RNG
    curandState localState = rngStates[tid];

    // Count the number of points which lie inside the unit quarter-circle
    unsigned int pointsInside = 0;

    for (unsigned int i = tid ; i < numSims ; i += step)
    {
        Real x;
        Real y;
        getPoint(x, y, localState);
        Real l2norm2 = x * x + y * y;

        if (l2norm2 < static_cast<Real>(1))
        {
            pointsInside++;
        }
    }

    // Reduce within the block
    pointsInside = reduce_sum(pointsInside);

    // Store the result
    if (threadIdx.x == 0)
    {
        results[bid] = pointsInside;
    }
}
