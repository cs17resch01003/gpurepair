//pass
//--local_size=16 --num_groups=2 --no-inline

#define tid get_local_id(0)

int foo(__local int* p) {
    return p[tid];
}

__kernel void baz(__local int* p) {
    foo(p);
    p[tid + 1] = tid;
}


