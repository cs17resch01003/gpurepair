//pass
//--local_size=64 --num_groups=64 --no-inline

char bar(void);

__kernel void foo()
{
  char x = bar();

  x = x + x;
  x++;

}
