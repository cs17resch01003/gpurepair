//xfail:ASSERTION_ERROR
//--local_size=1024 --num_groups=1024 --no-inline

float* bar(float* p, int i)
{
  if (i)
    return p;
  else
    return NULL;
}

__kernel void foo(int i)
{
  float x = 0;
  float *y = bar(&x, i);
  y[0] = y[0] + 1;
}
