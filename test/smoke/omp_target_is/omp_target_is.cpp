#include <iostream>
#include <omp.h>

// The test is meant to run in both USM and non-USM mode.

#if defined(__OFFLOAD_ARCH_gfx90a__) || defined(__OFFLOAD_ARCH_gfx90c__) || defined(__OFFLOAD_ARCH_gfx940__) || defined(__OFFLOAD_ARCH_gfx941__) || defined(__OFFLOAD_ARCH_gfx942__) || defined(__OFFLOAD_ARCH_gfx1010__) || defined(__OFFLOAD_ARCH_gfx1011__) || defined(__OFFLOAD_ARCH_gfx1012__) || defined(__OFFLOAD_ARCH_gfx1013__)
#define IS_USM 1
#else
#define IS_USM 0
#endif

#if IS_USM >=1
#pragma omp requires unified_shared_memory
#endif

#define N 1000

struct TT {
  int *p;
};

int main() {
  int *a = (int *) malloc(2*sizeof(int));
  a[0] = 1;
  a[1] = 2;
  int b[10];
  int *c = (int *) malloc(2*sizeof(int));
  TT *t = new TT;
  int n = N;
  t->p = (int *) malloc(n*sizeof(int));

#if IS_USM >= 1
  printf("\n ---- THIS IS USM ----\n");
#endif
  int is_set = 1;
  bool success = false;
  int rc_a_0, rc_a_1, rc_b, rc_c, rc_t, rc_t_p;

  rc_a_0 = omp_target_is_present(a, omp_get_default_device());
  rc_a_1 = omp_target_is_present(&a[1], omp_get_default_device());
  rc_b = omp_target_is_present(b, omp_get_default_device());
  rc_c = omp_target_is_present(c, omp_get_default_device());
  rc_t = omp_target_is_present(t, omp_get_default_device());
  rc_t_p = omp_target_is_present(t->p, omp_get_default_device());
  fprintf(stderr," a0,a1,b,c,t,t->p is present before target data region   : %d,%d,%d,%d,%d,%d \n", rc_a_0, rc_a_1, rc_b, rc_c, rc_t, rc_t_p);
#if IS_USM >= 1
  // omp_target_is_present returns false under USM mode regardless of whether a pointer has been mapped or not
  success = !rc_a_0;
  success = success && !rc_a_1;
  success = success && !rc_b;
  success = success && !rc_c;
  success = success && !rc_t;
  success = success && !rc_t_p;
#else
  // nothing has been mapped yet
  success = !rc_a_0;
  success = success && !rc_a_1;
  success = success && !rc_b;
  success = success && !rc_c;
  success = success && !rc_t;
  success = success && !rc_t_p;
#endif

#pragma omp target enter data map(to: a[0:1], c[1:1], t, t->p[:n])
  fprintf(stderr,"HOST   : a:%p &a[1]:%p a[0]:%d  a[1]:%d\n", (void*) a, &a[1],a[0],a[1]);
  rc_a_0 = omp_target_is_present(a,omp_get_default_device());
  rc_a_1 = omp_target_is_present(&a[1],omp_get_default_device());
  rc_b = omp_target_is_present(b, omp_get_default_device());
  rc_c = omp_target_is_present(c, omp_get_default_device());
  rc_t = omp_target_is_present(t, omp_get_default_device());
  rc_t_p = omp_target_is_present(t->p, omp_get_default_device());
  fprintf(stderr," a0,a1,b,c,t,t->p is present before target region   : %d,%d,%d,%d,%d,%d \n", rc_a_0, rc_a_1, rc_b, rc_c, rc_t, rc_t_p);
#if IS_USM >= 1
  // omp_target_is_present returns false under USM mode regardless of whether a pointer has been mapped or not
  success = success && !rc_a_0;
  success = success && !rc_a_1;
  success = success && !rc_b;
  success = success && !rc_c;
  success = success && !rc_t;
  success = success && !rc_t_p;
#else
  // a[0], p, p->t are mapped, then we expect is_present to return true (!0)
  success = success && rc_a_0;
  success = success && !rc_a_1;
  success = success && !rc_b;
  success = success && !rc_c; // only &c[1] is mapped
  success = success && rc_t;
  success = success && rc_t_p;
#endif

#pragma omp target map(tofrom: is_set)
  {
#if IS_USM >= 1
    // We use proprocessing macro, because cannot yet reference data in USM mode, even in an unused printf
    printf("DEVICE : a:%p &a[1]:%p No VALUES in USM MODE!\n", (void*) a,(void*) &a[1]);
#else
    printf("DEVICE : a:%p &a[1]:%p a[0]:%d  a[1]:%d\n", (void*) a, &a[1],a[0],a[1]);
    // is_set is a host pointer with xnack- and not accessible on the device
    if (a == nullptr)
      is_set = 0;
    else
      is_set = 1;
#endif
  }

  rc_a_0 = omp_target_is_present(a,omp_get_default_device());
  rc_a_1 = omp_target_is_present(&a[1],omp_get_default_device());
  rc_b = omp_target_is_present(b, omp_get_default_device());
  rc_c = omp_target_is_present(c, omp_get_default_device());
  rc_t = omp_target_is_present(t, omp_get_default_device());
  rc_t_p = omp_target_is_present(t->p, omp_get_default_device());
  fprintf(stderr," a0,a1,b,c,t,t->p is present inside target region   : %d,%d,%d,%d,%d,%d \n", rc_a_0, rc_a_1, rc_b, rc_c, rc_t, rc_t_p);
#if IS_USM >= 1
  // omp_target_is_present returns false under USM mode regardless of whether a pointer has been mapped or not
  success = success && !rc_a_0;
  success = success && !rc_a_1;
  success = success && !rc_b;
  success = success && !rc_c;
  success = success && !rc_t;
  success = success && !rc_t_p;
#else
  // a[0], p, p->t are mapped, then we expect is_present to return true (!0)
  success = success && rc_a_0;
  success = success && !rc_a_1;
  success = success && !rc_b;
  success = success && !rc_c; // only &c[1] is mapped
  success = success && rc_t;
  success = success && rc_t_p;
#endif

#pragma omp target exit data map(delete:a[0:1], c, t, t->p[:n])
  rc_a_0 = omp_target_is_present(a,omp_get_default_device());
  rc_a_1 = omp_target_is_present(&a[1],omp_get_default_device());
  rc_b = omp_target_is_present(b, omp_get_default_device());
  rc_c = omp_target_is_present(c, omp_get_default_device());
  rc_t = omp_target_is_present(t, omp_get_default_device());
  rc_t_p = omp_target_is_present(t->p, omp_get_default_device());
  fprintf(stderr," a0,a1,b,c,t,t->p is present outside target region   : %d,%d,%d,%d,%d,%d \n", rc_a_0, rc_a_1, rc_b, rc_c, rc_t, rc_t_p);

#if IS_USM >= 1
  // omp_target_is_present returns false under USM mode regardless of whether a pointer has been mapped or not
  success = success && !rc_a_0;
  success = success && !rc_a_1;
  success = success && !rc_b;
    success = success && !rc_c;
#else
  // a[0], t, t->p map are removed at the end of the target data region, then we expect is_present to return false (0)
  success = success && !rc_a_0;
  success = success && !rc_a_1;
  success = success && !rc_b;
  success = success && !rc_c;
  success = success && !rc_t;
  success = success && !rc_t_p;
#endif

  if(!success)
    return 1;

  return 0;
}
