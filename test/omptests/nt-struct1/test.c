
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (992)

typedef struct S {
  double a, b, c;
  double A[N], B[N];
  double d, e, f;
  double *pC;
  double x, y, z;
  long valC;
} S;

void print(S *s) 
{
  printf("  a %5.3f, b %5.3f, c %5.3f\n", s->a,  s->b, s->c);
  printf("  d %5.3f, e %5.3f, f %5.3f\n", s->d,  s->e, s->f);
  printf("  x %5.3f, y %5.3f, z %5.3f\n", s->x,  s->y, s->z);
}

#define ALLOC_STRUCT(s) { s.pC = malloc(N*sizeof(double)); }
#define FREE_STRUCT(s) { free(s.pC); }

#define INIT_STRUCT(s) { \
  s.a=0; s.b=1; s.c=2; \
  s.d=3; s.e=4; s.f=5; \
  s.x=6; s.y=7; s.z=8; \
  INIT_LOOP(N, {s.A[i] = 1; s.B[i] = i; s.pC[i] = -i; })}

#define VERIFY1(var, val) { if (var != val) { \
      printf("error with %s, expected %f, got %f\n", #var, var, val); fail++; \
  }} 

#define VERIFY_STRUCT(s, aa, bb, cc, dd, ee, ff, xx, yy, zz, AA, BB, CC) {\
  VERIFY1(s.a, 0.0+(aa)) \
  VERIFY1(s.b, 1.0+(bb)) \
  VERIFY1(s.c, 2.0+(cc)) \
  VERIFY1(s.d, 3.0+(dd)) \
  VERIFY1(s.e, 4.0+(ee)) \
  VERIFY1(s.f, 5.0+(ff)) \
  VERIFY1(s.x, 6.0+(xx)) \
  VERIFY1(s.y, 7.0+(yy)) \
  VERIFY1(s.z, 8.0+(zz)) \
  VERIFY(0, N, s.A[i], 1.0)                   \
  VERIFY(0, N, s.B[i], (double)i)             \
  VERIFY(0, N, s.pC[i], (double)-i)           \
}

#define ZERO(X) ZERO_ARRAY(N, X) 

#define MAP_ALL 0
#define MAP_SOME1 1

int main(void) {
  //check_offloading();

  S s1;

  ALLOC_STRUCT(s1);

#if MAP_ALL
  printf("Map all\n");
  INIT_STRUCT(s1);
  print(&s1);
  TEST_MAP(
    INIT_STRUCT(s1), 
    _Pragma("omp target map(s1)"), 
    { s1.a++; s1.c++; s1.e++; s1.x++; }, 
    VERIFY_STRUCT(s1, 1,0,1,  0,1,0,  1,0,0,  0,0,0));
  print(&s1);
#endif

#if MAP_SOME1
  printf("Map partial\n");
  INIT_STRUCT(s1);
  print(&s1);
  TEST_MAP(
    INIT_STRUCT(s1), 
    _Pragma("omp target map(s1.a, s1.b)"), 
    { s1.a++; s1.b++; }, 
    VERIFY_STRUCT(s1, 1,1,0,  0,0,0,  0,0,0,  0,0,0));
  print(&s1);
#endif

  return 0;
}
