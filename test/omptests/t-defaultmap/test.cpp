#include <stdio.h>
#include <omp.h>

#define V(x,y) \
  if ((x) != (y))\
    printf("Error %d != %d\n", (x), (y));\
  else\
    printf("Success!\n");
#define VPTR(x,y) \
  if ((x) != (y))\
    printf("Error %p != %p\n", (x), (y));\
  else\
    printf("Success!\n");

int A;

int main(void) {
  int Old = 1;
  int New = Old + 1;
  
  int *OldPtr = &A;
  int *NewPtr = OldPtr + 1;

  A = Old;
  int B = Old;
  int C1 = Old;
  int &C = C1;
  int *D = OldPtr;
  int *E1 = OldPtr;
  int *&E = E1;

  #pragma omp target device(1) defaultmap(tofrom: scalar) map(to:New, NewPtr)
  {
    A = New;
    B = New;
    C = New;
    D = NewPtr;
    E = NewPtr;
  }
  
  V(A, New);
  V(B, New);
  V(C, New);
  VPTR(D, NewPtr);
  VPTR(E, NewPtr);
  
  return 0;
}
