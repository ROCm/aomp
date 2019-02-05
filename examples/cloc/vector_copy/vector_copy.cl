__kernel void vector_copy(__global int *in, __global int *out) {
  int id = get_global_id(0);
  out[id] = in[id];
}
