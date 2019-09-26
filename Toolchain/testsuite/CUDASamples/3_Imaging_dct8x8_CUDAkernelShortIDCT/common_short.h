__device__ static __attribute__((always_inline)) short unfixh(int x);
__device__ static __attribute__((always_inline)) int unfixo(int x);
__device__ static __attribute__((always_inline)) void CUDAshortInplaceDCT(short *SrcDst, int Stride);
__device__ static __attribute__((always_inline)) void CUDAshortInplaceDCT(unsigned int *V8);
__device__ static __attribute__((always_inline)) void CUDAshortInplaceIDCT(short *SrcDst, int Stride);
__device__ static __attribute__((always_inline)) void CUDAshortInplaceIDCT(unsigned int *V8);

#define BLOCK_SIZE 8

#define KERS_BLOCK_WIDTH            32
#define KERS_BLOCK_HEIGHT           32
#define KERS_BW_LOG2                5
#define KERS_BH_LOG2                5
#define KERS_SMEMBLOCK_STRIDE       (KERS_BLOCK_WIDTH + 2)
#define KERS_BLOCK_WIDTH_HALF       (KERS_BLOCK_WIDTH / 2)

#define SIN_1_4     0x5A82
#define COS_1_4     0x5A82
#define SIN_1_8     0x30FC
#define COS_1_8     0x7642

#define OSIN_1_16   0x063E
#define OSIN_3_16   0x11C7
#define OSIN_5_16   0x1A9B
#define OSIN_7_16   0x1F63

#define OCOS_1_16   0x1F63
#define OCOS_3_16   0x1A9B
#define OCOS_5_16   0x11C7
#define OCOS_7_16   0x063E

#define FMUL(x,y) ((x)*(y))

union PackedShorts
{
    struct __align__(8)
    {
        short hShort1;
        short hShort2;
    };
    unsigned int hInt;
};

__device__ static __attribute__((always_inline)) short unfixh(int x)
{
    return (short)((x + 0x8000) >> 16);
}

__device__ static __attribute__((always_inline)) int unfixo(int x)
{
    return (x + 0x1000) >> 13;
}

__device__ static __attribute__((always_inline)) void CUDAshortInplaceDCT(short *SrcDst, int Stride)
{
    int in0, in1, in2, in3, in4, in5, in6, in7;
    int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    int tmp10, tmp11, tmp12, tmp13;
    int tmp14, tmp15, tmp16, tmp17;
    int tmp25, tmp26;

    int DoubleStride = Stride << 1;

    short *DstPtr = SrcDst;
    in0 = *DstPtr;
    DstPtr += Stride;
    in1 = *DstPtr;
    DstPtr += Stride;
    in2 = *DstPtr;
    DstPtr += Stride;
    in3 = *DstPtr;
    DstPtr += Stride;
    in4 = *DstPtr;
    DstPtr += Stride;
    in5 = *DstPtr;
    DstPtr += Stride;
    in6 = *DstPtr;
    DstPtr += Stride;
    in7 = *DstPtr;

    tmp0 = in7 + in0;
    tmp1 = in6 + in1;
    tmp2 = in5 + in2;
    tmp3 = in4 + in3;
    tmp4 = in3 - in4;
    tmp5 = in2 - in5;
    tmp6 = in1 - in6;
    tmp7 = in0 - in7;

    tmp10 = tmp3 + tmp0;
    tmp11 = tmp2 + tmp1;
    tmp12 = tmp1 - tmp2;
    tmp13 = tmp0 - tmp3;

    tmp16 = unfixo(FMUL(tmp6 + tmp5, SIN_1_4));
    tmp15 = unfixo(FMUL(tmp6 - tmp5, COS_1_4));

    tmp4 <<= 2;
    tmp7 <<= 2;

    tmp14 = tmp4 + tmp15;
    tmp25 = tmp4 - tmp15;
    tmp26 = tmp7 - tmp16;
    tmp17 = tmp7 + tmp16;

    DstPtr = SrcDst;
    *DstPtr = unfixh(FMUL(tmp10 + tmp11, SIN_1_4));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp13, COS_1_8) + FMUL(tmp12, SIN_1_8));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp10 - tmp11, COS_1_4));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp13, SIN_1_8) - FMUL(tmp12, COS_1_8));

    DstPtr = SrcDst + Stride;
    *DstPtr = unfixh(FMUL(tmp17, OCOS_1_16) + FMUL(tmp14, OSIN_1_16));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp26, OCOS_3_16) - FMUL(tmp25, OSIN_3_16));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp26, OCOS_5_16) + FMUL(tmp25, OSIN_5_16));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp17, OCOS_7_16) - FMUL(tmp14, OSIN_7_16));
}

__device__ static __attribute__((always_inline)) void CUDAshortInplaceDCT(unsigned int *V8)
{
    int in0, in1, in2, in3, in4, in5, in6, in7;
    int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    int tmp10, tmp11, tmp12, tmp13;
    int tmp14, tmp15, tmp16, tmp17;
    int tmp25, tmp26;
    PackedShorts sh0, sh1, sh2, sh3;

    sh0.hInt = V8[0];
    sh1.hInt = V8[1];
    sh2.hInt = V8[2];
    sh3.hInt = V8[3];
    in0 = sh0.hShort1;
    in1 = sh0.hShort2;
    in2 = sh1.hShort1;
    in3 = sh1.hShort2;
    in4 = sh2.hShort1;
    in5 = sh2.hShort2;
    in6 = sh3.hShort1;
    in7 = sh3.hShort2;

    tmp0 = in7 + in0;
    tmp1 = in6 + in1;
    tmp2 = in5 + in2;
    tmp3 = in4 + in3;
    tmp4 = in3 - in4;
    tmp5 = in2 - in5;
    tmp6 = in1 - in6;
    tmp7 = in0 - in7;

    tmp10 = tmp3 + tmp0;
    tmp11 = tmp2 + tmp1;
    tmp12 = tmp1 - tmp2;
    tmp13 = tmp0 - tmp3;

    sh0.hShort1 = unfixh(FMUL(tmp10 + tmp11, SIN_1_4));
    sh2.hShort1 = unfixh(FMUL(tmp10 - tmp11, COS_1_4));

    sh1.hShort1 = unfixh(FMUL(tmp13, COS_1_8) + FMUL(tmp12, SIN_1_8));
    sh3.hShort1 = unfixh(FMUL(tmp13, SIN_1_8) - FMUL(tmp12, COS_1_8));

    tmp16 = unfixo(FMUL(tmp6 + tmp5, SIN_1_4));
    tmp15 = unfixo(FMUL(tmp6 - tmp5, COS_1_4));

    tmp4 <<= 2;
    tmp7 <<= 2;

    tmp14 = tmp4 + tmp15;
    tmp25 = tmp4 - tmp15;
    tmp26 = tmp7 - tmp16;
    tmp17 = tmp7 + tmp16;

    sh0.hShort2 = unfixh(FMUL(tmp17, OCOS_1_16) + FMUL(tmp14, OSIN_1_16));
    sh3.hShort2 = unfixh(FMUL(tmp17, OCOS_7_16) - FMUL(tmp14, OSIN_7_16));
    sh2.hShort2 = unfixh(FMUL(tmp26, OCOS_5_16) + FMUL(tmp25, OSIN_5_16));
    sh1.hShort2 = unfixh(FMUL(tmp26, OCOS_3_16) - FMUL(tmp25, OSIN_3_16));

    V8[0] = sh0.hInt;
    V8[1] = sh1.hInt;
    V8[2] = sh2.hInt;
    V8[3] = sh3.hInt;
}

__device__ static __attribute__((always_inline)) void CUDAshortInplaceIDCT(short *SrcDst, int Stride)
{
    int in0, in1, in2, in3, in4, in5, in6, in7;
    int tmp10, tmp11, tmp12, tmp13;
    int tmp20, tmp21, tmp22, tmp23;
    int tmp30, tmp31;
    int tmp40, tmp41, tmp42, tmp43;
    int tmp50, tmp51, tmp52, tmp53;

    short *DstPtr = SrcDst;
    in0 = *DstPtr;
    DstPtr += Stride;
    in1 = *DstPtr;
    DstPtr += Stride;
    in2 = *DstPtr;
    DstPtr += Stride;
    in3 = *DstPtr;
    DstPtr += Stride;
    in4 = *DstPtr;
    DstPtr += Stride;
    in5 = *DstPtr;
    DstPtr += Stride;
    in6 = *DstPtr;
    DstPtr += Stride;
    in7 = *DstPtr;

    tmp10 = FMUL(in0 + in4, COS_1_4);
    tmp11 = FMUL(in0 - in4, COS_1_4);
    tmp12 = FMUL(in2, SIN_1_8) - FMUL(in6, COS_1_8);
    tmp13 = FMUL(in6, SIN_1_8) + FMUL(in2, COS_1_8);

    tmp20 = tmp10 + tmp13;
    tmp21 = tmp11 + tmp12;
    tmp22 = tmp11 - tmp12;
    tmp23 = tmp10 - tmp13;

    tmp30 = unfixo(FMUL(in3 + in5, COS_1_4));
    tmp31 = unfixo(FMUL(in3 - in5, COS_1_4));

    in1 <<= 2;
    in7 <<= 2;

    tmp40 = in1 + tmp30;
    tmp41 = in7 + tmp31;
    tmp42 = in1 - tmp30;
    tmp43 = in7 - tmp31;

    tmp50 = FMUL(tmp40, OCOS_1_16) + FMUL(tmp41, OSIN_1_16);
    tmp51 = FMUL(tmp40, OSIN_1_16) - FMUL(tmp41, OCOS_1_16);
    tmp52 = FMUL(tmp42, OCOS_5_16) + FMUL(tmp43, OSIN_5_16);
    tmp53 = FMUL(tmp42, OSIN_5_16) - FMUL(tmp43, OCOS_5_16);

    DstPtr = SrcDst;
    *DstPtr = unfixh(tmp20 + tmp50);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp21 + tmp53);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp22 + tmp52);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp23 + tmp51);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp23 - tmp51);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp22 - tmp52);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp21 - tmp53);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp20 - tmp50);
}

__device__ static __attribute__((always_inline)) void CUDAshortInplaceIDCT(unsigned int *V8)
{
    int in0, in1, in2, in3, in4, in5, in6, in7;
    int tmp10, tmp11, tmp12, tmp13;
    int tmp20, tmp21, tmp22, tmp23;
    int tmp30, tmp31;
    int tmp40, tmp41, tmp42, tmp43;
    int tmp50, tmp51, tmp52, tmp53;
    PackedShorts sh0, sh1, sh2, sh3;

    sh0.hInt = V8[0];
    sh1.hInt = V8[1];
    sh2.hInt = V8[2];
    sh3.hInt = V8[3];
    in0 = sh0.hShort1;
    in1 = sh0.hShort2;
    in2 = sh1.hShort1;
    in3 = sh1.hShort2;
    in4 = sh2.hShort1;
    in5 = sh2.hShort2;
    in6 = sh3.hShort1;
    in7 = sh3.hShort2;

    tmp10 = FMUL(in0 + in4, COS_1_4);
    tmp11 = FMUL(in0 - in4, COS_1_4);
    tmp12 = FMUL(in2, SIN_1_8) - FMUL(in6, COS_1_8);
    tmp13 = FMUL(in6, SIN_1_8) + FMUL(in2, COS_1_8);

    tmp20 = tmp10 + tmp13;
    tmp21 = tmp11 + tmp12;
    tmp22 = tmp11 - tmp12;
    tmp23 = tmp10 - tmp13;

    tmp30 = unfixo(FMUL(in3 + in5, COS_1_4));
    tmp31 = unfixo(FMUL(in3 - in5, COS_1_4));

    in1 <<= 2;
    in7 <<= 2;

    tmp40 = in1 + tmp30;
    tmp41 = in7 + tmp31;
    tmp42 = in1 - tmp30;
    tmp43 = in7 - tmp31;

    tmp50 = FMUL(tmp40, OCOS_1_16) + FMUL(tmp41, OSIN_1_16);
    tmp51 = FMUL(tmp40, OSIN_1_16) - FMUL(tmp41, OCOS_1_16);
    tmp52 = FMUL(tmp42, OCOS_5_16) + FMUL(tmp43, OSIN_5_16);
    tmp53 = FMUL(tmp42, OSIN_5_16) - FMUL(tmp43, OCOS_5_16);

    sh0.hShort1 = unfixh(tmp20 + tmp50);
    sh0.hShort2 = unfixh(tmp21 + tmp53);
    sh1.hShort1 = unfixh(tmp22 + tmp52);
    sh1.hShort2 = unfixh(tmp23 + tmp51);
    sh2.hShort1 = unfixh(tmp23 - tmp51);
    sh2.hShort2 = unfixh(tmp22 - tmp52);
    sh3.hShort1 = unfixh(tmp21 - tmp53);
    sh3.hShort2 = unfixh(tmp20 - tmp50);

    V8[0] = sh0.hInt;
    V8[1] = sh1.hInt;
    V8[2] = sh2.hInt;
    V8[3] = sh3.hInt;
}

#define IMAD(a, b, c) ( ((a) * (b)) + (c) )
#define IMUL(a, b) ((a) * (b))
