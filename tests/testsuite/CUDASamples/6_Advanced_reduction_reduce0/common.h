template<class T>
struct SharedMemory
{
    __device__ __attribute__((always_inline)) operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ __attribute__((always_inline)) operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ __attribute__((always_inline)) operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ __attribute__((always_inline)) operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};
