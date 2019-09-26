//pass
// --gridDim=128 --blockDim=256 --warp-sync=32

//REQUIRES: macros to make invariants more readable
#define ENABLED_OUTER (vec < vectorN)
#define ENABLED_INNER (stride != 0)

#define IMUL(a, b) __mul24(a, b)
#define ACCUM_N 1024

__global__ void scalarProdGPU(
    float *d_C,
    float *d_A,
    float *d_B,
    int vectorN,
    int elementN
)
{
    __requires(vectorN == (1 << 16));
    __requires(elementN == (1 << 16));
    
    //Accumulators cache
    __shared__ float accumResult[ACCUM_N];

    ////////////////////////////////////////////////////////////////////////////
    // Cycle through every pair of vectors,
    // taking into account that vector counts can be different
    // from total number of thread blocks
    ////////////////////////////////////////////////////////////////////////////
    for (int vec = blockIdx.x;         __global_invariant(__write_implies(d_C, threadIdx.x == 0)),         __global_invariant(__read_implies(accumResult, threadIdx.x == 0)),         __global_invariant(__read_implies(accumResult, (__read_offset_bytes(accumResult)/sizeof(float) == 0) | (__read_offset_bytes(accumResult)/sizeof(float) == 1))),         vec < vectorN; vec += gridDim.x)      
    {
        int vectorBase = IMUL(elementN, vec);
        int vectorEnd  = vectorBase + elementN;

        ////////////////////////////////////////////////////////////////////////
        // Each accumulator cycles through vectors with
        // stride equal to number of total number of accumulators ACCUM_N
        // At this stage ACCUM_N is only preferred be a multiple of warp size
        // to meet memory coalescing alignment constraints.
        ////////////////////////////////////////////////////////////////////////
        for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x)
        {
            float sum = 0;

            for (int pos = vectorBase + iAccum; pos < vectorEnd; pos += ACCUM_N){
                sum += d_A[pos] * d_B[pos];
            }

            accumResult[iAccum] = sum;
        }

        ////////////////////////////////////////////////////////////////////////
        // Perform tree-like reduction of accumulators' results.
        // ACCUM_N has to be power of two at this stage
        ////////////////////////////////////////////////////////////////////////
        for (int stride = ACCUM_N / 2;             __invariant((stride & (stride - 1)) == 0),             __global_invariant(__implies(ENABLED_OUTER, 0 <= stride & stride <= 512)),             __global_invariant(__implies(ENABLED_OUTER & !ENABLED_INNER & __read(accumResult), threadIdx.x == 0)),             __global_invariant(__implies(ENABLED_OUTER & !ENABLED_INNER & __read(accumResult), (__read_offset_bytes(accumResult)/sizeof(float) == 0) | (__read_offset_bytes(accumResult)/sizeof(float) == 1))),             stride > 0; stride >>= 1)
                {
            // __syncthreads();

            for (int iAccum = threadIdx.x;                 __global_invariant(__implies(ENABLED_OUTER & ENABLED_INNER, __read_implies(accumResult, (((__read_offset_bytes(accumResult)/sizeof(float)) % blockDim.x) == threadIdx.x) | (((__read_offset_bytes(accumResult)/sizeof(float) - stride) % blockDim.x) == threadIdx.x)))),                 __global_invariant(__implies(ENABLED_OUTER & ENABLED_INNER, __read_implies(accumResult, (__read_offset_bytes(accumResult)/sizeof(float)) < 2*stride))),                 __global_invariant(__implies(ENABLED_OUTER & ENABLED_INNER, __write_implies(accumResult, threadIdx.x < stride))),                 __global_invariant(__implies(ENABLED_OUTER & ENABLED_INNER, __read_implies(accumResult, threadIdx.x < stride))),                 __global_invariant(__implies(ENABLED_OUTER & !ENABLED_INNER & __read(accumResult), threadIdx.x == 0)),                 __global_invariant(__implies(ENABLED_OUTER & !ENABLED_INNER & __read(accumResult), (__read_offset_bytes(accumResult)/sizeof(float) == 0) | (__read_offset_bytes(accumResult)/sizeof(float) == 1))),                 iAccum < stride; iAccum += blockDim.x) 
            {
                accumResult[iAccum] += accumResult[stride + iAccum];
            }
        }
        
        if (threadIdx.x == 0){
         d_C[vec] = accumResult[0];
        }

    }
}
