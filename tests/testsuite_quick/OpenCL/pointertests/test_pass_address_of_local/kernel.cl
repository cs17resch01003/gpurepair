//xfail:ASSERTION_ERROR
//--local_size=64 --num_groups=64 --no-inline


void bar(__private int* x)
{
  *x = 5;
}

__kernel void foo()
{
  int x;

  x = 4;

  bar(&x);

  __assert(x == 4);

}

