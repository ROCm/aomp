extern "C" __global__ void vector_copy(int *in, int *out) {
  int id = (blockIdx.x*blockDim.x) + threadIdx.x;
  out[id] = in[id];
}
