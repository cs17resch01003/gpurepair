//pass
//--gridDim=256 --blockDim=256

__global__ void shfl_scan_test(int *data, int width, int *partial_sums=NULL)
{
    extern __shared__ int sums[];
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    // determine a warp_id within a block
    int warp_id = threadIdx.x / warpSize;

    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
    int value = data[id];

    // Now accumulate in log steps up the chain
    // compute sums, with another thread's value who is
    // distance delta away (i).  Note
    // those threads where the thread 'i' away would have
    // been out of bounds of the warp are unaffected.  This
    // creates the scan sum.
#pragma unroll

    for (int i=1; i<=width; i*=2)
    {
        int n = __shfl_up(value, i, width);

        if (lane_id >= i){
         value += n;
        }
    }

    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp

    // write the sum of the warp to smem
    if (threadIdx.x % warpSize == warpSize-1)
    {
        sums[warp_id] = value;
    }

__syncthreads();
    // __syncthreads();
// 
    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (warp_id == 0)
    {
        int warp_sum = sums[lane_id];

        for (int i=1; i<=width; i*=2)
        {
            int n = __shfl_up(warp_sum, i, width);

            if (lane_id >= i){
             warp_sum += n;
            }
        }

        sums[lane_id] = warp_sum;
    }

    // __syncthreads();

    // perform a uniform add across warps in the block
    // read neighbouring warp's sum and add it to threads value
    int blockSum = 0;

__syncthreads();
    if (warp_id > 0)
    {
        blockSum = sums[warp_id-1];
    }

    value += blockSum;

    // Now write out our result
    data[id] = value;

    // last thread has sum, write write out the block's sum
    if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
    {
        partial_sums[blockIdx.x] = value;
    }
}
