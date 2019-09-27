//xfail:NOT_ALL_VERIFIED
//--gridDim=1                --blockDim=32

typedef unsigned int clock_t;
#define syncthreads __syncthreads

__global__ void sum(clock_t *d_clocks, int N)
{
    __shared__ clock_t s_clocks[32];

    clock_t my_sum = 0;

    for (int i = threadIdx.x; i < N; i+= blockDim.x)
    {
        my_sum += d_clocks[i];
    }

    s_clocks[threadIdx.x] = my_sum;
    // syncthreads();

    for (int i=16; i>0; i/=2)
    {
        if (threadIdx.x < i)
        {
            int tmp= s_clocks[threadIdx.x + i];
            s_clocks[threadIdx.x] += tmp;
        }

        // syncthreads();
    }

    d_clocks[0] = s_clocks[0];
}
