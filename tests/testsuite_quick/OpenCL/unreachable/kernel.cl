//xfail:ASSERTION_ERROR
//--local_size=1024 --num_groups=1 --no-inline

void f(__global int *a) __attribute__ ((noreturn));

__kernel
void k(__global int *a)
{
  f(a);
}
