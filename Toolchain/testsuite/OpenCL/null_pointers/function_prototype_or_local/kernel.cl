//xfail:TIMEOUT
//--local_size=1024 --num_groups=1024 --no-inline

float* bar(float* p);

__kernel void foo(int i)
{
  float x = 0;
  float z = 0;
  float *y;

  if (i == 0)
    y = bar(&x);
  else
    y = &z;

  y[0] = y[0] + 1;
}
