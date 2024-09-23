#include <stdio.h>
#include <chrono>
#include <string>
#include <cmath>
#include <omp.h>
#include <stdlib.h>

// Timing infrastructure
using namespace std;

struct timer {
  const char *func;
  using clock_ty = std::chrono::high_resolution_clock;
  std::chrono::time_point<clock_ty> start, next;

  explicit timer(const char *func): func(func) {
    start = clock_ty::now();
    next = start;
  }

  void checkpoint(const char *func) {
    auto end = clock_ty::now();

    uint64_t t =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - next)
            .count();

    printf("%35s: %lu ns (%f ms)\n", func, t, t / 1e6);
    next = clock_ty::now();
  }

  ~timer() {
    auto end = clock_ty::now();

    uint64_t t =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    printf("%35s: %lu ns (%f ms)\n", func, t, t / 1e6);
  }
};

int vecadd(long size, std::string label) {
  timer stopwatch(label.c_str());
  printf("Size = %ld\n", size);
  double *a;
  double *b;
  double *c;
  double *d;
  double *e;
  double *f;
  double *g;
  double *h;
  double *i;
  double *j;
  double *k;
  double *l;
  double *m;
  double *n;
  double *o;
  double *p;
  double *q;
  double *r;
  double *s;
  double *t;

  posix_memalign((void **)&a, 4096 ,size*sizeof(double));
  posix_memalign((void **)&b, 4096 ,size*sizeof(double));
  posix_memalign((void **)&c, 4096 ,size*sizeof(double));
  posix_memalign((void **)&d, 4096 ,size*sizeof(double));
  posix_memalign((void **)&e, 4096 ,size*sizeof(double));
  posix_memalign((void **)&f, 4096 ,size*sizeof(double));
  posix_memalign((void **)&g, 4096 ,size*sizeof(double));
  posix_memalign((void **)&h, 4096 ,size*sizeof(double));
  posix_memalign((void **)&i, 4096 ,size*sizeof(double));
  posix_memalign((void **)&j, 4096 ,size*sizeof(double));
  posix_memalign((void **)&k, 4096 ,size*sizeof(double));
  posix_memalign((void **)&l, 4096 ,size*sizeof(double));
  posix_memalign((void **)&m, 4096 ,size*sizeof(double));
  posix_memalign((void **)&n, 4096 ,size*sizeof(double));
  posix_memalign((void **)&o, 4096 ,size*sizeof(double));
  posix_memalign((void **)&p, 4096 ,size*sizeof(double));
  posix_memalign((void **)&q, 4096 ,size*sizeof(double));
  posix_memalign((void **)&r, 4096 ,size*sizeof(double));
  posix_memalign((void **)&s, 4096 ,size*sizeof(double));
  posix_memalign((void **)&t, 4096 ,size*sizeof(double));

  for (long it = 0; it < size; it++) {
    a[it] = (double)0;
    b[it] = (double)it;
    c[it] = (double)it;
    d[it] = (double)it;
    e[it] = (double)it;
    f[it] = (double)it;
    g[it] = (double)it;
    h[it] = (double)it;
    i[it] = (double)it;
    j[it] = (double)it;
    k[it] = (double)it;
    l[it] = (double)it;
    m[it] = (double)it;
    n[it] = (double)it;
    o[it] = (double)it;
    p[it] = (double)it;
    q[it] = (double)it;
    r[it] = (double)it;
    s[it] = (double)it;
    t[it] = (double)it;
  }

   #pragma omp target enter data map(alloc: a[:size], b[:size],c[:size],d[:size],e[:size],f[:size],g[:size],h[:size], \
   i[:size],j[:size],k[:size],l[:size],m[:size],n[:size],o[:size],p[:size], \
     q[:size],r[:size],s[:size],t[:size])

  string initLab = label+" Init and alloc";
  stopwatch.checkpoint(initLab.c_str());

  for(int is = 0; is < 8; is++) {
    if (is == 4) {
      string warmupLab = label+" kernel warmup";
      stopwatch.checkpoint(warmupLab.c_str());
    }
      #pragma omp target teams distribute parallel for			\
	map(always, to:b[:size],c[:size],d[:size],e[:size],f[:size],g[:size],h[:size], \
	    i[:size],j[:size],k[:size],l[:size],m[:size],n[:size],o[:size],p[:size], \
	    q[:size],r[:size],s[:size],t[:size]) map(always, from:a[:size])
      for (long it = 0; it < size; it++)
	a[it] = b[it]+c[it]+d[it]+e[it]+f[it]+g[it]+h[it]+
	  i[it]+j[it]+k[it]+l[it]+m[it]+n[it]+o[it]+p[it]+
	  q[it]+r[it]+s[it]+t[it];
      if(is >= 4)  {
	string tgtLab = label+" target it = "+to_string(is);
	stopwatch.checkpoint(tgtLab.c_str());
      }
      // check results after each kernel
      int tmp_rc = 0;
      for (long it = 0; it < size; it++)
	if (a[it] != (double)(19*it)) {
	  tmp_rc++;
	  printf ("Wrong varlue: a[%ld]=%lf (exp: %lf)\n", it, a[it], (double)(19*it));
	  if (tmp_rc > 10) return tmp_rc;
	}
  }
  string tgtLab = label+" Target";
  stopwatch.checkpoint(tgtLab.c_str());
  int rc = 0;
  for (long it = 0; it < size; it++)
    if (a[it] != (double)(19*it)) {
      rc++;
      printf ("Wrong varlue: a[%ld]=%lf (exp: %lf)\n", it, a[it], (double)(19*it));
      if (rc > 10) break;
    }
  if (!rc) {
    string checkLab = label+" Check";
    stopwatch.checkpoint(checkLab.c_str());
  }

  #pragma omp target exit data map(delete: a, b,c, d, e, f, g, h,  \
   i, j, k, l, m, n, o, p,  \
     q, r, s, t)

  free(a);
  free(b);
  free(c);
  free(d);
  free(e);
  free(f);
  free(g);
  free(h);
  free(i);
  free(j);
  free(k);
  free(l);
  free(m);
  free(n);
  free(o);
  free(p);
  free(q);
  free(r);
  free(s);
  free(t);

  string deallocLab = label+" Dealloc";
  stopwatch.checkpoint(deallocLab.c_str());

  return rc;
}

int checkSize(long arraySize , float memSize, int initialExponent){
  int exponent = 0;
  float GiB =  pow(1024,3);
  float sizeGiB;
  int numArrays = 20;

  for (int i = initialExponent; i > 0; i--){
    sizeGiB = (arraySize * numArrays * i) / GiB;
    if (sizeGiB < memSize){
      exponent = i;
      return exponent;
   }
  }
  return initialExponent;
}

int main(int argc, char * argv[])
{
  int largeExponent = 8;
  int adjustedExponent = 0;
  long smallSize = pow(10,3);
  long largeSize = pow(10,largeExponent);

  // Expects size of GPU memory in GiB (Bytes/1024^3)
  if (argc > 1){
    float memSize = atof(argv[1]);
    adjustedExponent = checkSize(largeSize, memSize, largeExponent);
    largeSize =  pow(10,adjustedExponent);
    largeExponent = adjustedExponent;
    printf("largeExponent: %d\n", adjustedExponent);
  }

  int rc = vecadd(smallSize, "small (10^3)");
  if (rc) return rc;
  std::string largeString = "large (10^";
  largeString = largeString + std::to_string(largeExponent) + ")";
  rc = vecadd(largeSize, largeString);

  if (adjustedExponent != 0)
    printf("Warning: Large exponent was adjusted due to insufficient GPU memory. largeSize: %s\n", largeString.c_str());

  return rc;
}
