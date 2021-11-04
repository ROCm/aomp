__kernel void large_array_copy(__global int* a, __global  int* b, int N) {
  uint tid = get_global_id(0);
  a[tid] = b[tid];
}
