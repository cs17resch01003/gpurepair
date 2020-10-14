//pass
//--local_size=16 --num_groups=2 --no-inline

#define tid get_local_id(0)

int jazz(__local int *x, __local int *y, __local int *z) {
    return
        x[tid] +
        y[tid + 1] +
        z[tid + 1];
}

int sim(int x, __local int * b) {
    jazz(b, b, b);
    return 0;
}

int bar(__local int* a) {
    return a[tid] + sim(a[tid + 2], a);
}

int foo(__local int* p) {
    bar(p);
    return p[tid];
}

__kernel void baz(__local int* p) {
    foo(p);
    p[tid + 1] = tid;
}
