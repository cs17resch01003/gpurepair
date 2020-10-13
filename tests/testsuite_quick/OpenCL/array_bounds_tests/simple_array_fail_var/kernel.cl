//xfail:ASSERTION_ERROR
//--local_size=8 --num_groups=8 --check-array-bounds

__kernel void foo() {
	local int L[20];
	int x = get_global_id(0);
	L[x] = x * x;
}
