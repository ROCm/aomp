#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <sys/time.h>

typedef struct {
  double x, y, z;
} Monomer;

typedef struct {
  struct {
    size_t size;
    Monomer *buf, *dev_buf;
  } all_Monomer;
} Allocator;

typedef struct {
  Allocator *alloc;
} Phase;

int main() {
  size_t n = 1600000000;
  Allocator alloc;
  Phase phase;
  Phase *const p = &phase;
  p->alloc = &alloc;

  p->alloc->all_Monomer.size = n;
  p->alloc->all_Monomer.buf = (Monomer *)malloc(n*sizeof(Monomer));
  p->alloc->all_Monomer.dev_buf = (Monomer *)omp_target_alloc(n*sizeof(Monomer), 0);
  omp_target_associate_ptr(p->alloc->all_Monomer.buf, p->alloc->all_Monomer.dev_buf, n*sizeof(Monomer), 0, 0);

  #pragma omp target enter data map(to: p[:1])
  #pragma omp target enter data map(to: p->alloc[:1])

  struct timeval t1, t2;
  double td;
  size_t bytes =
      sizeof(p->alloc->all_Monomer.buf[0]) * p->alloc->all_Monomer.size;
  printf("Preparing to update to() at file: %s line: %d: %lu bytes\n", __FILE__,
         __LINE__, bytes);
  gettimeofday(&t1, 0);

  #pragma omp target update to(p->alloc->all_Monomer.buf[:p->alloc->all_Monomer.size])

  gettimeofday(&t2, 0);
  td = (t2.tv_sec + t2.tv_usec / 1e6) - (t1.tv_sec + t1.tv_usec / 1e6);
  printf("Update to(): %lu bytes %g seconds %g MB/sec\n", bytes, td,
         1e-6 * bytes / td);

  return 0;
}
