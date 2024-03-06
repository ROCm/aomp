#include<cstdint>
#include<cstdio>

// eager_maps: XNACK enabled and OMPX_EAGER_ZERO_COPY_MAPS=1, compiled with: gfx94X, without USM pragma => OUTCOME: AUTO ZERO-COPY with prefaulting

int main() {
  const int n = 1024*20;
  int64_t *a = new int64_t[n];
  int64_t *b = new int64_t[n];

  // init a and b
  for (int64_t i = 0; i < n; i++) {
    a[i] = -1;
    b[i] = i;
  }

  // CHECK: __tgt_rtl_prepopulate_page_table:  {{.+}} 163840
  #pragma omp target teams loop map(tofrom: a[:n], b[:n]) 
  for (int64_t i = 0; i < n; i++)
    a[i] += b[i];

  // check
  int err = 0;
  for (int64_t i = 0; i < n; i++)
    if (a[i] != b[i]-1) {
      err++;
      printf("Error at %ld: expected %ld, got %ld\n", i, b[i]-1, a[i]);
      if (err > 10) return err;
    }
  return err;
}
