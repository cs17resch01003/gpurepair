//xfail:NOT_ALL_VERIFIED
//--blockDim=64 --gridDim=64 --no-inline
//

#include "cuda.h"

__global__ void foo (int* p, int* q, int* r){

    int a, b, c;
    int* d;

    a = 10;

    d = &a;
    d = &b;

    if (a > 10)
    {
        d = &c;
    }
    else
    {
        p = q;
        q = p;
    }

    d[1] = 200;

    p[100] = q[100] + 1;

}
