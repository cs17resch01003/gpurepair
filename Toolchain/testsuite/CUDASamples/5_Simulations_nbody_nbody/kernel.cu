//xfail:NOT_ALL_VERIFIED
//--gridDim=[56,1,1] --blockDim=[256,1,1]

template<class T>
struct SharedMemory
{
    __device__ __attribute__((always_inline)) inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ __attribute__((always_inline)) inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template<typename T>
__device__ static __attribute__((always_inline)) T rsqrt_T(T x)
{
    return rsqrt(x);
}

template<>
__device__ static __attribute__((always_inline)) float rsqrt_T<float>(float x)
{
    return rsqrtf(x);
}

template <typename T> struct vec3
{
    typedef float   Type;
}; // dummy
template <>           struct vec3<float>
{
    typedef float3  Type;
};
template <>           struct vec3<double>
{
    typedef double3 Type;
};

template <typename T> struct vec4
{
    typedef float   Type;
}; // dummy
template <>           struct vec4<float>
{
    typedef float4  Type;
};
template <>           struct vec4<double>
{
    typedef double4 Type;
};

__constant__ float softeningSquared;
__constant__ double softeningSquared_fp64;

template <typename T>
__device__ static __attribute__((always_inline)) T getSofteningSquared()
{
    return softeningSquared;
}
template <>
__device__ static __attribute__((always_inline)) double getSofteningSquared<double>()
{
    return softeningSquared_fp64;
}

// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]

template <typename T>
__device__ static __attribute__((always_inline)) typename vec3<T>::Type
bodyBodyInteraction(typename vec3<T>::Type ai,
                    typename vec4<T>::Type bi,
                    typename vec4<T>::Type bj)
{
    typename vec3<T>::Type r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    T distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += getSofteningSquared<T>();

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist = rsqrt_T(distSqr);
    T invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}


// This is the "tile_calculation" function from the GPUG3 article.
template <typename T>
__device__ static __attribute__((always_inline)) typename vec3<T>::Type
gravitation(typename vec4<T>::Type iPos,
            typename vec3<T>::Type accel)
{
    typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();

    // The CUDA 1.1 compiler cannot determine that i is not going to
    // overflow in the loop below.  Therefore if int is used on 64-bit linux
    // or windows (or long instead of long long on win64), the compiler
    // generates suboptimal code.  Therefore we use long long on win64 and
    // long on everything else. (Workaround for Bug ID 347697)
#ifdef _Win64
    unsigned long long j = 0;
#else
    unsigned long j = 0;
#endif

    // Here we unroll the loop to reduce bookkeeping instruction overhead
    // 32x unrolling seems to provide best performance

    // Note that having an unsigned int loop counter and an unsigned
    // long index helps the compiler generate efficient code on 64-bit
    // OSes.  The compiler can't assume the 64-bit index won't overflow
    // so it incurs extra integer operations.  This is a standard issue
    // in porting 32-bit code to 64-bit OSes.

#pragma unroll 32

    for (unsigned int counter = 0; counter < blockDim.x; counter++)
    {
        accel = bodyBodyInteraction<T>(accel, iPos, SX(j++));
    }

    return accel;
}

// WRAP is used to force each block to start working on a different
// chunk (and wrap around back to the beginning of the array) so that
// not all multiprocessors try to read the same memory locations at
// once.
#define WRAP(x,m) (((x)<(m))?(x):((x)-(m)))  // Mod without divide, works on values from 0 up to 2m

template <typename T, bool multithreadBodies>
__device__ static __attribute__((always_inline)) typename vec3<T>::Type
computeBodyAccel(typename vec4<T>::Type bodyPos,
                 typename vec4<T>::Type *positions,
                 int numBodies)
{
    typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();

    typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};

    int p = blockDim.x;
    int q = blockDim.y;
    int n = numBodies;
    int numTiles = n / (p * q);

    for (int tile = blockIdx.y; tile < numTiles + blockIdx.y; tile++)
    {
        sharedPos[threadIdx.x+blockDim.x*threadIdx.y] =
            multithreadBodies ?
            positions[WRAP(blockIdx.x + q * tile + threadIdx.y, gridDim.x) * p + threadIdx.x] :
            positions[WRAP(blockIdx.x + tile,                   gridDim.x) * p + threadIdx.x];

        // __syncthreads();

        // This is the "tile_calculation" function from the GPUG3 article.
        acc = gravitation<T>(bodyPos, acc);

        // __syncthreads();
    }

    // When the numBodies / thread block size is < # multiprocessors (16 on G80), the GPU is
    // underutilized.  For example, with a 256 threads per block and 1024 bodies, there will only
    // be 4 thread blocks, so the GPU will only be 25% utilized. To improve this, we use multiple
    // threads per body.  We still can use blocks of 256 threads, but they are arranged in q rows
    // of p threads each.  Each thread processes 1/q of the forces that affect each body, and then
    // 1/q of the threads (those with threadIdx.y==0) add up the partial sums from the other
    // threads for that body.  To enable this, use the "--p=" and "--q=" command line options to
    // this example. e.g.: "nbody.exe --n=1024 --p=64 --q=4" will use 4 threads per body and 256
    // threads per block. There will be n/p = 16 blocks, so a G80 GPU will be 100% utilized.

    // We use a bool template parameter to specify when the number of threads per body is greater
    // than one, so that when it is not we don't have to execute the more complex code required!
    if (multithreadBodies)
    {
        SX_SUM(threadIdx.x, threadIdx.y).x = acc.x;
        SX_SUM(threadIdx.x, threadIdx.y).y = acc.y;
        SX_SUM(threadIdx.x, threadIdx.y).z = acc.z;

        // __syncthreads();

        // Save the result in global memory for the integration step
        if (threadIdx.y == 0)
        {
            for (int i = 1; i < blockDim.y; i++)
            {
                acc.x += SX_SUM(threadIdx.x,i).x;
                acc.y += SX_SUM(threadIdx.x,i).y;
                acc.z += SX_SUM(threadIdx.x,i).z;
            }
        }
    }

    return acc;
}

template<typename T, bool multithreadBodies>
__global__ void
integrateBodies(typename vec4<T>::Type *newPos,
                typename vec4<T>::Type *oldPos,
                typename vec4<T>::Type *vel,
                unsigned int deviceOffset, unsigned int deviceNumBodies,
                float deltaTime, float damping, int totalNumBodies);
template
__global__ void
integrateBodies<float, false>(
                vec4<float>::Type *newPos,
                vec4<float>::Type *oldPos,
                vec4<float>::Type *vel,
                unsigned int deviceOffset, unsigned int deviceNumBodies,
                float deltaTime, float damping, int totalNumBodies);


template<typename T, bool multithreadBodies>
__global__ void
integrateBodies(typename vec4<T>::Type *newPos,
                typename vec4<T>::Type *oldPos,
                typename vec4<T>::Type *vel,
                unsigned int deviceOffset, unsigned int deviceNumBodies,
                float deltaTime, float damping, int totalNumBodies)
{
    __requires(deviceNumBodies == 54*256); // ALLY: This has to be a multiple of blockDim.x, otherwise the kernel has barrier divergence.  To make things interesting I have made it slightly less than the total number of threads so that some thread blocks do not execute.
#ifdef FORCE_FAIL
    __ensures(false);
#endif
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= deviceNumBodies)
    {
        return;
    }

    typename vec4<T>::Type position = oldPos[deviceOffset + index];

    typename vec3<T>::Type accel = computeBodyAccel<T, multithreadBodies>(position, oldPos, totalNumBodies);


    if (!multithreadBodies || (threadIdx.y == 0))
    {
        // acceleration = force \ mass;
        // new velocity = old velocity + acceleration * deltaTime
        // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
        // (because they cancel out).  Thus here force == acceleration
        typename vec4<T>::Type velocity = vel[deviceOffset + index];

        velocity.x += accel.x * deltaTime;
        velocity.y += accel.y * deltaTime;
        velocity.z += accel.z * deltaTime;

        velocity.x *= damping;
        velocity.y *= damping;
        velocity.z *= damping;

        // new position = old position + velocity * deltaTime
        position.x += velocity.x * deltaTime;
        position.y += velocity.y * deltaTime;
        position.z += velocity.z * deltaTime;

        // store new position and velocity
        newPos[deviceOffset + index] = position;
        vel[deviceOffset + index]    = velocity;
    }
}
