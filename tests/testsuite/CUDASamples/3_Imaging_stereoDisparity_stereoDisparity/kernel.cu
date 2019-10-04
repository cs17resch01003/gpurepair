//xfail:NOT_ALL_VERIFIED
//--gridDim=[20,67] --blockDim=[32,8]

#define blockSize_x 32
#define blockSize_y 8

// RAD is the radius of the region of support for the search
#define RAD 8
// STEPS is the number of loads we must perform to initialize the shared memory area
// (see convolution SDK sample for example)
#define STEPS 3

texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex2Dleft;
texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex2Dright;

__device__ static __attribute__((always_inline)) unsigned int __usad4(unsigned int A, unsigned int B, unsigned int C=0);
/* IMPERIAL EDIT: inline asm commented out
{
    unsigned int result;
#if (__CUDA_ARCH__ >= 300) // Kepler (SM 3.x) supports a 4 vector SAD SIMD
    asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;": "=r"(result):"r"(A), "r"(B), "r"(C));
#else // SM 2.0            // Fermi  (SM 2.x) supports only 1 SAD SIMD, so there are 4 instructions
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b0, %2.b0, %3;": "=r"(result):"r"(A), "r"(B), "r"(C));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b1, %2.b1, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b2, %2.b2, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b3, %2.b3, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
#endif
    return result;
}
*/

__global__ void
stereoDisparityKernel(unsigned int *g_img0, unsigned int *g_img1,
                      unsigned int *g_odata,
                      int w, int h,
                      int minDisparity, int maxDisparity)
{
    __requires(w == 640);

    // access thread id
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int sidx = threadIdx.x+RAD;
    const unsigned int sidy = threadIdx.y+RAD;

    unsigned int imLeft;
    unsigned int imRight;
    unsigned int cost;
    unsigned int bestCost = 9999999;
    unsigned int bestDisparity = 0;
    __shared__ unsigned int diff[blockSize_y+2*RAD][blockSize_x+2*RAD];

    // store needed values for left image into registers (constant indexed local vars)
    unsigned int imLeftA[STEPS];
    unsigned int imLeftB[STEPS];

    for (int i=0; i<STEPS; i++)
    {
        int offset = -RAD + i*RAD;
        imLeftA[i] = tex2D(tex2Dleft, tidx-RAD, tidy+offset);
        imLeftB[i] = tex2D(tex2Dleft, tidx-RAD+blockSize_x, tidy+offset);
    }

    // for a fixed camera system this could be hardcoded and loop unrolled
    for (int d=minDisparity; d<=maxDisparity; d++)
    {
        //LEFT
#pragma unroll
        for (int i=0;             __global_invariant(__write_implies(diff, __write_offset_bytes(diff)/sizeof(unsigned int)%(blockSize_x + 2 * RAD) == sidx - RAD)),             __global_invariant(__write_implies(diff, (__write_offset_bytes(diff)/sizeof(unsigned int)/(blockSize_x + 2 * RAD) - sidy + RAD)%RAD == 0)),             i<STEPS; i++)        
        {
            int offset = -RAD + i*RAD;
            //imLeft = tex2D( tex2Dleft, tidx-RAD, tidy+offset );
            imLeft = imLeftA[i];
            imRight = tex2D(tex2Dright, tidx-RAD+d, tidy+offset);
            cost = __usad4(imLeft, imRight);
            diff[sidy+offset][sidx-RAD] = cost;
        }

        //RIGHT
#pragma unroll

        for (int i=0;             __global_invariant(__write_implies(diff, (__write_offset_bytes(diff)/sizeof(unsigned int)%(blockSize_x + 2 * RAD) == sidx - RAD + blockSize_x)                                                    | (__write_offset_bytes(diff)/sizeof(unsigned int)%(blockSize_x + 2 * RAD) == sidx - RAD))),             __global_invariant(__write_implies(diff, (__write_offset_bytes(diff)/sizeof(unsigned int)/(blockSize_x + 2 * RAD) - sidy + RAD)%RAD == 0)),             i<STEPS; i++)        
        {
            int offset = -RAD + i*RAD;

            if (threadIdx.x < 2*RAD)
            {
                //imLeft = tex2D( tex2Dleft, tidx-RAD+blockSize_x, tidy+offset );
                imLeft = imLeftB[i];
                imRight = tex2D(tex2Dright, tidx-RAD+blockSize_x+d, tidy+offset);
                cost = __usad4(imLeft, imRight);
                diff[sidy+offset][sidx-RAD+blockSize_x] = cost;
            }
        }

        // __syncthreads();

        // sum cost horizontally
#pragma unroll

        for (int j=0; j<STEPS; j++)
        {
            int offset = -RAD + j*RAD;
            cost = 0;
#pragma unroll

            for (int i=-RAD; i<=RAD ; i++)
            {
                cost += diff[sidy+offset][sidx+i];
            }

            // __syncthreads();
            diff[sidy+offset][sidx] = cost;
            // __syncthreads();

        }

        // sum cost vertically
        cost = 0;
#pragma unroll

        for (int i=-RAD; i<=RAD ; i++)
        {
            cost += diff[sidy+i][sidx];
        }

        // see if it is better or not
        if (cost < bestCost)
        {
            bestCost = cost;
            bestDisparity = d+8;
        }

        // __syncthreads();

    }

    if (tidy < h && tidx < w)
    {
        g_odata[tidy*w + tidx] = bestDisparity;
    }
}
