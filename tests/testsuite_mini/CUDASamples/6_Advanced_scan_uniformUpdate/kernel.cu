//pass
//--gridDim=6624 --blockDim=256

typedef unsigned int uint;

__global__ void uniformUpdate(
    uint4 *d_Data,
    uint *d_Buffer
)
{
    __shared__ uint buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0)
    {
        buf = d_Buffer[blockIdx.x];
    }

    // __syncthreads();

    uint4 data4 = d_Data[pos];
    data4.x += buf;
    data4.y += buf;
    data4.z += buf;
    data4.w += buf;
    d_Data[pos] = data4;
}
