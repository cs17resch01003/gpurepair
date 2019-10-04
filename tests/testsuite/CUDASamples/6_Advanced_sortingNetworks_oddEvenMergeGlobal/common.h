typedef unsigned int uint;
#define SHARED_SIZE_LIMIT 1024U
#define    UMUL(a, b) __umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

__device__ static __attribute__((always_inline)) void Comparator(
    uint &keyA,
    uint &valA,
    uint &keyB,
    uint &valB,
    uint dir
)
{
    uint t;

    if ((keyA > keyB) == dir)
    {
        t = keyA;
        keyA = keyB;
        keyB = t;
        t = valA;
        valA = valB;
        valB = t;
    }
}
