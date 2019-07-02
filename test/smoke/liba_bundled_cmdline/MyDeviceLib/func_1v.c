void func_1v(float* in, float* out, unsigned n){
  unsigned i;
  #pragma omp target teams distribute parallel for map(to: in[0:n]) map(from: out[0:n])
  for(i=0; i<n; ++i){
    out[i]=2*in[i];
  }
}