//pass
//--local_size=1024 --num_groups=2 --no-inline

__kernel void atomicTest(__local int *A, int B)
{
   A[get_local_id(0) + 1] = 42;
   __local char *C = (__local char*)A;
   atomic_add((__local int*)(C + 1), B);
}
