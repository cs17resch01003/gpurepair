//xfail:TIMEOUT
//--gridDim=[10,40]      --blockDim=[32,6]

template<int bx, int by> __global__ void JacobiIteration(const float *du0, const float *dv0, const float *Ix, const float *Iy, const float *Iz, int w, int h, int s, float alpha, float *du1, float *dv1);
template __global__ void JacobiIteration<32,6>(const float *du0, const float *dv0, const float *Ix, const float *Iy, const float *Iz, int w, int h, int s, float alpha, float *du1, float *dv1);
#define min(x,y) (x < y ? x : y)
#define max(x,y) (x < y ? y : x)

template<int bx, int by>
__global__
void JacobiIteration(const float *du0,
                     const float *dv0,
                     const float *Ix,
                     const float *Iy,
                     const float *Iz,
                     int w, int h, int s,
                     float alpha,
                     float *du1,
                     float *dv1)
{
    __requires(w == 320);
    __requires(h == 240);
    __requires(s == 320);
    volatile __shared__ float du[(bx + 2) * (by + 2)];
    volatile __shared__ float dv[(bx + 2) * (by + 2)];

    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // position within global memory array
    const int pos = min(ix, w - 1) + min(iy, h - 1) * s;

    // position within shared memory array
    const int shMemPos = threadIdx.x + 1 + (threadIdx.y + 1) * (bx + 2);

    // Load data to shared memory.
    // load tile being processed
    du[shMemPos] = du0[pos];
    dv[shMemPos] = dv0[pos];

    // load necessary neigbouring elements
    // We clamp out-of-range coordinates.
    // It is equivalent to mirroring
    // because we access data only one step away from borders.
    if (threadIdx.y == 0)
    {
        // beginning of the tile
        const int bsx = blockIdx.x * blockDim.x;
        const int bsy = blockIdx.y * blockDim.y;
        // element position within matrix
        int x, y;
        // element position within linear array
        // gm - global memory
        // sm - shared memory
        int gmPos, smPos;

        x = min(bsx + threadIdx.x, w - 1);
        // row just below the tile
        y = max(bsy - 1, 0);
        gmPos = y * s + x;
        smPos = threadIdx.x + 1;
        du[smPos] = du0[gmPos];
        dv[smPos] = dv0[gmPos];

        // row above the tile
        y = min(bsy + by, h - 1);
        smPos += (by + 1) * (bx + 2);
        gmPos  = y * s + x;
        du[smPos] = du0[gmPos];
        dv[smPos] = dv0[gmPos];
    }
    if (threadIdx.y == 1)
    {
        // beginning of the tile
        const int bsx = blockIdx.x * blockDim.x;
        const int bsy = blockIdx.y * blockDim.y;
        // element position within matrix
        int x, y;
        // element position within linear array
        // gm - global memory
        // sm - shared memory
        int gmPos, smPos;

        y = min(bsy + threadIdx.x, h - 1);
        // column to the left
        x = max(bsx - 1, 0);
        smPos = bx + 2 + threadIdx.x * (bx + 2);
        gmPos = x + y * s;

        // check if we are within tile
        if (threadIdx.x < by)
        {
            du[smPos] = du0[gmPos];
            dv[smPos] = dv0[gmPos];
            // column to the right
            x = min(bsx + bx, w - 1);
            gmPos  = y * s + x;
            smPos += bx + 1;
            du[smPos] = du0[gmPos];
            dv[smPos] = dv0[gmPos];
        }
    }

    // __syncthreads();

    if (ix >= w || iy >= h)
    {
     return;
    }

    // now all necessary data are loaded to shared memory
    int left, right, up, down;
    left  = shMemPos - 1;
    right = shMemPos + 1;
    up    = shMemPos + bx + 2;
    down  = shMemPos - bx - 2;

    float sumU = (du[left] + du[right] + du[up] + du[down]) * 0.25f;
    float sumV = (dv[left] + dv[right] + dv[up] + dv[down]) * 0.25f;

    float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos])
                 / (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

    du1[pos] = sumU - Ix[pos] * frac;
    dv1[pos] = sumV - Iy[pos] * frac;
}
