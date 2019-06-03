//xfail:ASSERTION_ERROR
//--blockDim=16 --gridDim=16 --no-inline

__global__ void example(unsigned a, unsigned b, unsigned c) {

    __requires(a == 12);
    __requires(b == 36);
    
    __assert(a + b != c);

}
