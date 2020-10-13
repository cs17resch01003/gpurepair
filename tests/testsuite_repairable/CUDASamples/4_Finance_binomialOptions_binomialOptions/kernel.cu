//xfail:TIMEOUT
//--gridDim=512 --blockDim=256

//REQUIRES: const array as formal (imperial edit)

#define min(x,y) (x < y ? x : y)

#ifndef DOUBLE_PRECISION
typedef float real;
#else
typedef double real;
#endif
#
//Number of time steps
#define   NUM_STEPS 2048
//Max option batch size
#define MAX_OPTIONS 1024

#define  TIME_STEPS 16
#define CACHE_DELTA (2 * TIME_STEPS)           
#define  CACHE_SIZE (256)
#define  CACHE_STEP (CACHE_SIZE - CACHE_DELTA)

#if NUM_STEPS % CACHE_DELTA
#error Bad constants
#endif

//Preprocessed input option data
typedef struct
{
    real S;
    real X;
    real vDt;
    real puByDf;
    real pdByDf;
} __TOptionData;

#if 0 // imperial edit
static __constant__ __TOptionData d_OptionData[MAX_OPTIONS];
static __device__           float d_CallValue[MAX_OPTIONS];
static __device__            real d_CallBuffer[MAX_OPTIONS * (NUM_STEPS + 16)];
#endif

////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
#ifndef DOUBLE_PRECISION
__device__ static __attribute__((always_inline)) float expiryCallValue(float S, float X, float vDt, int i)
{
    real d = S * expf(vDt * (2.0f * i - NUM_STEPS)) - X;
    return (d > 0) ? d : 0;
}
#else
__device__ static __attribute__((always_inline)) double expiryCallValue(double S, double X, double vDt, int i)
{
    double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
    return (d > 0) ? d : 0;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// GPU kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void binomialOptionsKernel(
  __TOptionData *d_OptionData, // imperial edit
  float *d_CallValue,          // imperial edit
  real *d_CallBuffer           // imperial edit
) {
    __shared__ real callA[CACHE_SIZE+1];
    __shared__ real callB[CACHE_SIZE+1];
    //Global memory frame for current option (thread block)
    real *const d_Call = &d_CallBuffer[blockIdx.x * (NUM_STEPS + 16)];

    const int       tid = threadIdx.x;
    const real      S = d_OptionData[blockIdx.x].S;
    const real      X = d_OptionData[blockIdx.x].X;
    const real    vDt = d_OptionData[blockIdx.x].vDt;
    const real puByDf = d_OptionData[blockIdx.x].puByDf;
    const real pdByDf = d_OptionData[blockIdx.x].pdByDf;

    //Compute values at expiry date
    for (int i = tid; i <= NUM_STEPS; i += CACHE_SIZE)
    {
        d_Call[i] = expiryCallValue(S, X, vDt, i);
    }

    //#if 0

    //Walk down binomial tree
    //So double-buffer and synchronize to avoid read-after-write hazards.
    for (int i = NUM_STEPS; i > 0; i -= CACHE_DELTA)
        for (int c_base = 0; c_base < i; c_base += CACHE_STEP)
        {
            //Start and end positions within shared memory cache
            int c_start = min(CACHE_SIZE - 1, i - c_base);
            int c_end   = c_start - CACHE_DELTA;

            //Read data(with apron) to shared memory
            // __syncthreads();

            if (tid <= c_start)
            {
              callA[tid] = d_Call[c_base + tid];
            }

            //Calculations within shared memory
            for (int k = c_start - 1; k >= c_end;)
            {
                //Compute discounted expected value
                // __syncthreads();
                callB[tid] = puByDf * callA[tid + 1] + pdByDf * callA[tid];
                k--;

                //Compute discounted expected value
                // __syncthreads();
                callA[tid] = puByDf * callB[tid + 1] + pdByDf * callB[tid];
                k--;
            }

            //Flush shared memory cache
            // __syncthreads();

            if (tid <= c_end)
            {
                d_Call[c_base + tid] = callA[tid];
            }
        }

    //Write the value at the top of the tree to destination buffer
    if (threadIdx.x == 0)
    {
        d_CallValue[blockIdx.x] = (float)callA[0];
    }

    //#endif

}
