#include <stdio.h>
#include <chrono>
#include <string>
#include <cmath>
#include <omp.h>



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
  double *a = new double[size];
  double *b = new double[size];
  double *c = new double[size];
  double *d = new double[size];
  double *e = new double[size];
  double *f = new double[size];
  double *g = new double[size];
  double *h = new double[size];
  double *i = new double[size];
  double *j = new double[size];
  double *k = new double[size];
  double *l = new double[size];
  double *m = new double[size];
  double *n = new double[size];
  double *o = new double[size];
  double *p = new double[size];
  double *q = new double[size];
  double *r = new double[size];
  double *s = new double[size];
  double *t = new double[size];

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

  #pragma omp target teams distribute parallel for	\
    map(always, to:b[:size],c[:size],d[:size],e[:size],f[:size],g[:size],h[:size], \
    i[:size],j[:size],k[:size],l[:size],m[:size],n[:size],o[:size],p[:size],	\
	q[:size],r[:size],s[:size],t[:size]) map(always,from:a[:size])
  for (long it = 0; it < size; it++)
      a[it] = b[it]+c[it]+d[it]+e[it]+f[it]+g[it]+h[it]+
	i[it]+j[it]+k[it]+l[it]+m[it]+n[it]+o[it]+p[it]+
	q[it]+r[it]+s[it]+t[it];

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

int main()
{
  long smallSize = pow(10,3);
  int rc = vecadd(smallSize, "small (10^3)");
  if (rc) return rc;
  
  long largeSize = pow(10,8);
  rc = vecadd(largeSize, "large (10^9)");

  return rc;
}
