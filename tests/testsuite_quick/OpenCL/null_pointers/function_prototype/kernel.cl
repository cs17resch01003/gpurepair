//xfail:ASSERTION_ERROR
//--local_size=1024 --num_groups=1024 --no-inline

float* bar(float* p);

__kernel void foo()
{
  float x = 0;
  float *y = bar(&x);
  y[0] = y[0] + 1;
}
