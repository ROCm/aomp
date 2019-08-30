void hfunc3(float* in, float* out, unsigned n){
  unsigned i;
  for(i=0; i<n; ++i){
    out[i]=in[i]+in[i];
  }
}
