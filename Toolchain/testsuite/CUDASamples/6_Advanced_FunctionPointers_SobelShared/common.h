// Texture reference for reading image
texture<unsigned char, 2> tex;
extern __shared__ unsigned char LocalBlock[];
#define LAST_BLOCK_FILTER 2
typedef unsigned char Pixel;

#define RADIUS 1

// pixel value used for thresholding function, works well with sample image 'lena'
#define THRESHOLD 150.0f

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

// A function pointer can be declared explicity like this line:
//__device__ unsigned char (*pointFunction)(unsigned char, float ) = NULL;
// or by using typedef's like below:

typedef unsigned char(*blockFunction_t)(
    unsigned char, unsigned char, unsigned char,
    unsigned char, unsigned char, unsigned char,
    unsigned char, unsigned char, unsigned char,
    float);

typedef unsigned char(*pointFunction_t)(
    unsigned char, float);

__device__ blockFunction_t blockFunction;

__device__ unsigned char
ComputeSobel(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale)
{
    short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
    short Vert = ul + 2*um + ur - ll - 2*lm - lr;
    short Sum = (short)(fScale*(abs((int)Horz)+abs((int)Vert)));
    return (unsigned char)((Sum < 0) ? 0 : ((Sum > 255) ? 255 : Sum)) ;
}

__device__ unsigned char
ComputeBox(unsigned char ul,   // upper left
           unsigned char um, // upper middle
           unsigned char ur, // upper right
           unsigned char ml, // middle left
           unsigned char mm, // middle...middle
           unsigned char mr, // middle right
           unsigned char ll, // lower left
           unsigned char lm, // lower middle
           unsigned char lr, // lower right
           float fscale
          )
{

    short Sum = (short)(ul+um+ur + ml+mm+mr + ll+lm+lr)/9;
    Sum *= fscale;
    return (unsigned char)((Sum < 0) ? 0 : ((Sum > 255) ? 255 : Sum)) ;
}

__device__ unsigned char
Threshold(unsigned char in, float thresh)
{
    if (in > thresh)
    {
        return 0xFF;
    }
    else
    {
        return 0;
    }
}

__device__ blockFunction_t blockFunction_table[LAST_BLOCK_FILTER];
