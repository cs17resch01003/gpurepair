//pass
//--gridDim=[32,32] --blockDim=[16,16]

__device__ static __attribute__((always_inline)) int rgbToInt(float r, float g, float b);
__device__ static __attribute__((always_inline)) uchar4 getPixel(int x, int y);

#ifndef USE_TEXTURE_RGBA8UI
texture<float4, 2, cudaReadModeElementType> inTex;
#else
texture<uchar4, 2, cudaReadModeElementType> inTex;
#endif

// convert floating point rgb color to 8-bit integer
__device__ static __attribute__((always_inline)) int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

// get pixel from 2D image, with clamping to border
__device__ static __attribute__((always_inline)) uchar4 getPixel(int x, int y)
{
#ifndef USE_TEXTURE_RGBA8UI
    float4 res = tex2D(inTex, x, y);
    uchar4 ucres = make_uchar4(res.x*255.0f, res.y*255.0f, res.z*255.0f, res.w*255.0f);
#else
    uchar4 ucres = tex2D(inTex, x, y);
#endif
    return ucres;
}

// macros to make indexing shared memory easier
#define SMEM(X, Y) sdata[(Y)*tilew+(X)]

__global__ void
cudaProcess(unsigned int *g_odata, int imgw, int imgh,
            int tilew, int r, float threshold, float highlight)
{
    __requires(imgw == 512);
    __requires(tilew == 32);
    __requires(r == 8);
    extern __shared__ uchar4 sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;

#if 0
    uchar4 c4 = getPixel(x, y);
    g_odata[y*imgw+x] = rgbToInt(c4.z, c4.y, c4.x);
#else
    // copy tile to shared memory
    // center region
    SMEM(r + tx, r + ty) = getPixel(x, y);

    // borders
    if (threadIdx.x < r)
    {
        // left
        SMEM(tx, r + ty) = getPixel(x - r, y);
        // right
        SMEM(r + bw + tx, r + ty) = getPixel(x + bw, y);
    }

    if (threadIdx.y < r)
    {
        // top
        SMEM(r + tx, ty) = getPixel(x, y - r);
        // bottom
        SMEM(r + tx, r + bh + ty) = getPixel(x, y + bh);
    }

    // load corners
    if ((threadIdx.x < r) && (threadIdx.y < r))
    {
        // tl
        SMEM(tx, ty) = getPixel(x - r, y - r);
        // bl
        SMEM(tx, r + bh + ty) = getPixel(x - r, y + bh);
        // tr
        SMEM(r + bw + tx, ty) = getPixel(x + bh, y - r);
        // br
        SMEM(r + bw + tx, r + bh + ty) = getPixel(x + bw, y + bh);
    }

    // wait for loads to complete
    // __syncthreads();

    // perform convolution
    float rsum = 0.0f;
    float gsum = 0.0f;
    float bsum = 0.0f;
    float samples = 0.0f;

    for (int dy=-r; dy<=r; dy++)
    {
        for (int dx=-r; dx<=r; dx++)
        {
#if 0
            // try this to see the benefit of using shared memory
            uchar4 pixel = getPixel(x+dx, y+dy);
#else
            uchar4 pixel = SMEM(r+tx+dx, r+ty+dy);
#endif

            // only sum pixels within disc-shaped kernel
            float l = dx*dx + dy*dy;

            if (l <= r*r)
            {
                float r = float(pixel.x);
                float g = float(pixel.y);
                float b = float(pixel.z);
#if 1
                // brighten highlights
                float lum = (r + g + b) / (255*3);

                if (lum > threshold)
                {
                    r *= highlight;
                    g *= highlight;
                    b *= highlight;
                }

#endif
                rsum += r;
                gsum += g;
                bsum += b;
                samples += 1.0f;
            }
        }
    }

    rsum /= samples;
    gsum /= samples;
    bsum /= samples;
    // ABGR
    g_odata[y*imgw+x] = rgbToInt(rsum, gsum, bsum);
    //g_odata[y*imgw+x] = rgbToInt(x,y,0);
#endif
}
