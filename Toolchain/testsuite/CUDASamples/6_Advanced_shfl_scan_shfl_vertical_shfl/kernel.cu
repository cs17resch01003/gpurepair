//xfail:NOT_ALL_VERIFIED
//--gridDim=[60,1] --blockDim=[32,8]

__global__ void shfl_vertical_shfl(unsigned int *img, int width, int height)
{
    __requires(width == 1920);

    __shared__ unsigned int sums[32][9];
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    //int warp_id = threadIdx.x / warpSize ;
    unsigned int lane_id = tidx % 8;
    //int rows_per_thread = (height / blockDim. y) ;
    //int start_row = rows_per_thread * threadIdx.y;
    unsigned int stepSum = 0;

    sums[threadIdx.x][threadIdx.y] = 0;
    // __syncthreads();

    for (int step = 0 ;         __global_invariant(__write_implies(img, (__write_offset_bytes(img)/sizeof(unsigned int))%width== tidx)),         __global_invariant(__read_implies(img, (__read_offset_bytes(img)/sizeof(unsigned int))%width == tidx)),         __global_invariant(__write_implies(img, ((__write_offset_bytes(img)/sizeof(unsigned int) - tidx)/width - threadIdx.y)%8 == 0)),         step < 135 ; step++)
    {
        unsigned int sum = 0;
        unsigned int *p = img + (threadIdx.y+step*8)*width + tidx;

        sum = *p;
        sums[threadIdx.x][threadIdx.y] = sum;
        // __syncthreads();

        // place into SMEM
        // shfl scan reduce the SMEM, reformating so the column
        // sums are computed in a warp
        // then read out properly
        int partial_sum = 0;
        int j = threadIdx.x %8;
        int k = threadIdx.x/8 + threadIdx.y*4;

        partial_sum = sums[k][j];

        for (int i=1 ; i<=8 ; i*=2)
        {
            int n = __shfl_up(partial_sum, i, 32);

            if (lane_id >= i){
             partial_sum += n;
            }
        }

        sums[k][j] = partial_sum;
        // __syncthreads();

        if (threadIdx.y > 0)
        {
            sum += sums[threadIdx.x][threadIdx.y-1];
        }

        sum += stepSum;
        stepSum += sums[threadIdx.x][blockDim.y-1];
        // __syncthreads();
        *p = sum ;
    }

}
