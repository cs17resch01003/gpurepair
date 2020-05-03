//pass
//--gridDim=255 --blockDim=256

__global__ void uniform_add(int *data, int *partial_sums, int len)
{
    __requires(len == 65536);

    __shared__ int buf;
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

    if (id > len){
     return;
    }

    if (threadIdx.x == 0)
    {
        buf = partial_sums[blockIdx.x];
    }

    // __syncthreads();
	__syncthreads();
    data[id] += buf;
}
