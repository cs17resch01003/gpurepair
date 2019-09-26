//pass
//--gridDim=4096 --blockDim=2048 3

__global__ void fwtBatch1Kernel(float *d_Output, float *d_Input, int log2N)
{
    __requires(log2N == 11);
    const int    N = 1 << log2N;
    const int base = blockIdx.x << log2N;

    //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
    extern __shared__ float s_data[];
    float *d_Src = d_Input  + base;
    float *d_Dst = d_Output + base;

    for (int pos = threadIdx.x; pos < N; pos += blockDim.x)
    {
        s_data[pos] = d_Src[pos];
    }

    //Main radix-4 stages
    const int pos = threadIdx.x;

    for (int stride = N >> 2; stride > 0; stride >>= 2)
    {
        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        // __syncthreads();
        float D0 = s_data[i0];
        float D1 = s_data[i1];
        float D2 = s_data[i2];
        float D3 = s_data[i3];

        float T;
        T = D0;
        D0         = D0 + D2;
        D2         = T - D2;
        T = D1;
        D1         = D1 + D3;
        D3         = T - D3;
        T = D0;
        s_data[i0] = D0 + D1;
        s_data[i1] = T - D1;
        T = D2;
        s_data[i2] = D2 + D3;
        s_data[i3] = T - D3;
    }

    //Do single radix-2 stage for odd power of two
    if (log2N & 1)
    {
        // __syncthreads();

        for (int pos = threadIdx.x;
             __global_invariant(__write_implies(s_data, (__write_offset_bytes(s_data)/sizeof(float)/2)%blockDim.x == threadIdx.x)),
             __global_invariant(__read_implies(s_data, (__read_offset_bytes(s_data)/sizeof(float)/2)%blockDim.x == threadIdx.x)),
             pos < N / 2; pos += blockDim.x)
        {
            int i0 = pos << 1;
            int i1 = i0 + 1;

            float D0 = s_data[i0];
            float D1 = s_data[i1];
            s_data[i0] = D0 + D1;
            s_data[i1] = D0 - D1;
        }
    }

    // __syncthreads();

    for (int pos = threadIdx.x;
         __global_invariant(__write_implies(d_Output, (__write_offset_bytes(d_Output)/sizeof(float) - base)%blockDim.x == threadIdx.x)),
         pos < N; pos += blockDim.x)
    {
        d_Dst[pos] = s_data[pos];
    }
}
