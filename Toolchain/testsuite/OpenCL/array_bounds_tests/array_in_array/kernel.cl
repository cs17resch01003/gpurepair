//xfail:ASSERTION_ERROR
//--local_size=8 --num_groups=8 --check-array-bounds

__kernel void foo() {
	local int L[64];
	local int G[64];
	L[G[get_global_id(0)]] = get_global_size(0);
}
