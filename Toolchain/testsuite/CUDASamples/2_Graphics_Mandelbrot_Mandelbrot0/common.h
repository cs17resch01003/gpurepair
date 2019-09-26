__device__ unsigned int blockCounter;   // global counter, initialized to zero before kernel launch

// The core Mandelbrot CUDA GPU calculation function
#if 1
// Unrolled version
template<class T>
__device__ static __attribute__((always_inline))
int CalcMandelbrot(const T xPos, const T yPos, const T xJParam, const T yJParam, const int crunch, const bool isJulia)
{
    T x, y, xx, yy ;
    int i = crunch;

    T xC, yC ;

    if (isJulia)
    {
        xC = xJParam ;
        yC = yJParam ;
        y = yPos;
        x = xPos;
        yy = y * y;
        xx = x * x;

    }
    else
    {
        xC = xPos ;
        yC = yPos ;
        y = 0 ;
        x = 0 ;
        yy = 0 ;
        xx = 0 ;
    }

    do
    {
        // Iteration 1
        if (xx + yy > T(4.0))
            return i - 1;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 2
        if (xx + yy > T(4.0))
            return i - 2;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 3
        if (xx + yy > T(4.0))
            return i - 3;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 4
        if (xx + yy > T(4.0))
            return i - 4;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 5
        if (xx + yy > T(4.0))
            return i - 5;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 6
        if (xx + yy > T(4.0))
            return i - 6;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 7
        if (xx + yy > T(4.0))
            return i - 7;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 8
        if (xx + yy > T(4.0))
            return i - 8;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 9
        if (xx + yy > T(4.0))
            return i - 9;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 10
        if (xx + yy > T(4.0))
            return i - 10;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 11
        if (xx + yy > T(4.0))
            return i - 11;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 12
        if (xx + yy > T(4.0))
            return i - 12;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 13
        if (xx + yy > T(4.0))
            return i - 13;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 14
        if (xx + yy > T(4.0))
            return i - 14;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 15
        if (xx + yy > T(4.0))
            return i - 15;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 16
        if (xx + yy > T(4.0))
            return i - 16;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 17
        if (xx + yy > T(4.0))
            return i - 17;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 18
        if (xx + yy > T(4.0))
            return i - 18;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 19
        if (xx + yy > T(4.0))
            return i - 19;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;

        // Iteration 20
        i -= 20;

        if ((i <= 0) || (xx + yy > T(4.0)))
            return i;

        y = x * y * T(2.0) + yC;
        x = xx - yy + xC;
        yy = y * y;
        xx = x * x;
    }
    while (1);
} // CalcMandelbrot

#else

template<class T>
__device__ static __attribute__((always_inline))
int CalcMandelbrot(const T xPos, const T yPos, const T xJParam, const T yJParam, const int crunch, const isJulia)
{
    T x, y, xx, yy, xC, yC ;

    if (isJulia)
    {
        xC = xJParam ;
        yC = yJParam ;
        y = yPos;
        x = xPos;
        yy = y * y;
        xx = x * x;

    }
    else
    {
        xC = xPos ;
        yC = yPos ;
        y = 0 ;
        x = 0 ;
        yy = 0 ;
        xx = 0 ;
    }

    int i = crunch;

    while (--i && (xx + yy < T(4.0)))
    {
        y = x * y * T(2.0) + yC ;
        x = xx - yy + xC ;
        yy = y * y;
        xx = x * x;
    }

    return i; // i > 0 ? crunch - i : 0;
} // CalcMandelbrot

#endif

#define ABS(n) ((n) < 0 ? -(n) : (n))

// Determine if two pixel colors are within tolerance
__device__ static __attribute__((always_inline)) int CheckColors(const uchar4 &color0, const uchar4 &color1)
{
    int x = color1.x - color0.x;
    int y = color1.y - color0.y;
    int z = color1.z - color0.z;
    return (ABS(x) > 10) || (ABS(y) > 10) || (ABS(z) > 10);
} // CheckColors
