//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//[\s]*kernel.cl:11:[\d]+:[\s]+error:[\s]+postcondition might not hold on all return paths[\s]+__ensures\(__implies\(__enabled\(\), __return_val_int\(\) > 0\)\);





int bar(int a) {
  __requires(__implies(__enabled(), a > 0));
  __ensures(__implies(__enabled(), __return_val_int() > 0));
  return a/2;
}

__kernel void foo() {

  int x, y;
  x = bar(5);
  y = bar(6);

}
