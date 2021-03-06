//xfail:NOT_ALL_VERIFIED
//--equality-abstraction --local_size=1024 --num_groups=2

kernel void foo(__global unsigned *A) {
    A[get_global_id(0)] = 1;

    for (int i = 0; i < 1024; ++i) {
        A[get_global_id(0)] = 1;
    }

    __assert(A[get_global_id(0)] == 1);

    __function_wide_invariant(A[get_global_id(0)] == 1);
}
