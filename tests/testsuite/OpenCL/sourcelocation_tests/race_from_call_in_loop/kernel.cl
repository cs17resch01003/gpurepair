//pass
//--local_size=16 --num_groups=2 --no-inline

#define tid get_local_id(0)

int foo(__local int * A) {
    return A[tid];
}

__kernel void bar(__local int * p) {

    for(int i = 0; i < 100; i++) {
        foo(p);
    }

    p[tid + 1] = tid;

}

