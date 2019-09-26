//pass
//--gridDim=[2,128] --blockDim=[16,4] --no-inline 2

#include "common.h"

__global__ void SobelShared(uchar4 *pSobelOriginal, unsigned short SobelPitch,
#ifndef FIXED_BLOCKWIDTH
            short BlockWidth, short SharedPitch,
#endif
            short w, short h, float fScale,
            int blockOperation, pointFunction_t pPointFunction
           )
{
    __requires(SobelPitch == 512);
    __requires(w == 512);
#ifndef FIXED_BLOCKWIDTH
    __requires(BlockWidth == 80);
    __requires(SharedPitch == 384);
#endif
    __requires(blockOperation == 0 | blockOperation == 1);
    __requires(blockFunction_table[0] == ComputeSobel);
    __requires(blockFunction_table[1] == ComputeBox);
    __requires(pPointFunction == Threshold | pPointFunction == NULL);

    short u = 4*blockIdx.x*BlockWidth;
    short v = blockIdx.y*blockDim.y + threadIdx.y;
    short ib;

    int SharedIdx = threadIdx.y * SharedPitch;

    for (ib = threadIdx.x;
         __global_invariant(ib%blockDim.x == threadIdx.x),
         __global_invariant(__write_implies(LocalBlock, (__write_offset_bytes(LocalBlock)-SharedIdx)/4 < (BlockWidth+2*RADIUS))),
         __global_invariant(__write_implies(LocalBlock, (__write_offset_bytes(LocalBlock)-SharedIdx)/4%blockDim.x == threadIdx.x)),
         ib < BlockWidth+2*RADIUS; ib += blockDim.x)
    {
        LocalBlock[SharedIdx+4*ib+0] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+0), (float)(v-RADIUS));
        LocalBlock[SharedIdx+4*ib+1] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+1), (float)(v-RADIUS));
        LocalBlock[SharedIdx+4*ib+2] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+2), (float)(v-RADIUS));
        LocalBlock[SharedIdx+4*ib+3] = tex2D(tex,
                                             (float)(u+4*ib-RADIUS+3), (float)(v-RADIUS));
    }

    if (threadIdx.y < RADIUS*2)
    {
        //
        // copy trailing RADIUS*2 rows of pixels into shared
        //
        SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;

        for (ib = threadIdx.x;
             #define SharedIdxOld (threadIdx.y * SharedPitch)
             __global_invariant(__implies(threadIdx.y < RADIUS*2, ib%blockDim.x == threadIdx.x)),
             __global_invariant(__implies(threadIdx.y < RADIUS*2, __write_implies(LocalBlock,
                  (((__write_offset_bytes(LocalBlock)-SharedIdx)/4 < (BlockWidth+2*RADIUS))
                   & ((__write_offset_bytes(LocalBlock)-SharedIdx)/4%blockDim.x == threadIdx.x))
                | (((__write_offset_bytes(LocalBlock)-SharedIdxOld)/4 < (BlockWidth+2*RADIUS))
                   & ((__write_offset_bytes(LocalBlock)-SharedIdxOld)/4%blockDim.x == threadIdx.x))))),
             ib < BlockWidth+2*RADIUS; ib += blockDim.x)
        {
            LocalBlock[SharedIdx+4*ib+0] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+0), (float)(v+blockDim.y-RADIUS));
            LocalBlock[SharedIdx+4*ib+1] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+1), (float)(v+blockDim.y-RADIUS));
            LocalBlock[SharedIdx+4*ib+2] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+2), (float)(v+blockDim.y-RADIUS));
            LocalBlock[SharedIdx+4*ib+3] = tex2D(tex,
                                                 (float)(u+4*ib-RADIUS+3), (float)(v+blockDim.y-RADIUS));
        }
    }

    // __syncthreads();

    u >>= 2;    // index as uchar4 from here
    uchar4 *pSobel = (uchar4 *)(((char *) pSobelOriginal)+v*SobelPitch);
    SharedIdx = threadIdx.y * SharedPitch;

    blockFunction = blockFunction_table[blockOperation];

    for (ib = threadIdx.x;
         __global_invariant(ib%blockDim.x == threadIdx.x),
         __global_invariant(__write_implies(pSobelOriginal, (((__write_offset_bytes(pSobelOriginal) - v*SobelPitch)/sizeof(uchar4)) - u)%blockDim.x == threadIdx.x)),
         __global_invariant(__write_implies(pSobelOriginal, (((__write_offset_bytes(pSobelOriginal) - v*SobelPitch)/sizeof(uchar4)) - u) < BlockWidth)),
         __global_invariant(__write_implies(pSobelOriginal, (__write_offset_bytes(pSobelOriginal) < (v + 1)*SobelPitch))),
         ib < BlockWidth; ib += blockDim.x)
    {

        uchar4 out;

        unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
        unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
        unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
        unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
        unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
        unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
        unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
        unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
        unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];

        out.x = (*blockFunction)(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale);

        pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
        pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
        pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
        out.y = (*blockFunction)(pix01, pix02, pix00,
                                 pix11, pix12, pix10,
                                 pix21, pix22, pix20, fScale);

        pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
        pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
        pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
        out.z = (*blockFunction)(pix02, pix00, pix01,
                                 pix12, pix10, pix11,
                                 pix22, pix20, pix21, fScale);

        pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
        pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
        pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
        out.w = (*blockFunction)(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale);

        if (pPointFunction != NULL)
        {
            out.x = (*pPointFunction)(out.x, THRESHOLD);
            out.y = (*pPointFunction)(out.y, THRESHOLD);
            out.z = (*pPointFunction)(out.z, THRESHOLD);
            out.w = (*pPointFunction)(out.w, THRESHOLD);
        }

        if (u+ib < w/4 && v < h)
        {
            pSobel[u+ib] = out;
        }

    }

    // __syncthreads();
}
