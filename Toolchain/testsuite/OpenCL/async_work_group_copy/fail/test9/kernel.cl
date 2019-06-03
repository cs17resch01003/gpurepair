//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float* __restrict p, global float * __restrict q) {

    local float my_p[N];
    local float my_q[N];
    
    event_t joint_handle;

    joint_handle = async_work_group_copy(my_p, p + N*get_group_id(0), N, 0);
    async_work_group_copy(my_q, q + N*get_group_id(0), N, joint_handle);

    // Error: reached non-uniformly
    if(get_local_id(0) > 13) {
        wait_group_events(1, &joint_handle);
    }

}
