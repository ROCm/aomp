#include <stdio.h>

#define M (1024*1024)
#define BUFF_SIZE (1*M)

#define N (8*BUFF_SIZE)

int b[N];

int Test(int start, int size)
{
  int i;
  int errors = 0;

  for(i=0; i<start; i++) b[i] = -1;
  for(i=start; i<size; i++) b[i] = i;
  for(i=size; i<N; i++) b[i] = -1;

  #pragma omp target parallel for
  {
    for(int i=start; i<size; i++) b[i] += 1;
  }

  for(i=0; i<start && errors<25; i++) {
    if (b[i] != -1) printf("%4i: before, got %d, expected %d, %d error\n", i, b[i], -1, ++errors);
  }
  for(i=start; i<size && errors<25; i++) {
    if (b[i] != i+1) printf("%4i: in, got %d, expected %d, %d error\n", i, b[i], i+1, ++errors);
  }
  for(i=size; i<N && errors<25; i++) {
    if (b[i] != -1) printf("%4i: after, got %d, expected %d, %d error\n", i, b[i], -1, ++errors);
  }

  if (errors>0) { 
    printf("success with start %d, size %d (%d mod buff size)\n\n", start, size, size % BUFF_SIZE);
  } else {
    printf("%d errors with start %d, size %d (%d mod buff size)\n\n", errors, start, size, size % BUFF_SIZE);
  }
  return (errors>0);
}

int main()
{
  int offset[] = {0, 1, 2, BUFF_SIZE/2, BUFF_SIZE-2, BUFF_SIZE-1};
  int onum = 6;
  int errors = 0;

  for(int s1=0; s1<6; s1++) {
    for(int s2=0; s2<6; s2++) {
      errors += Test(offset[s1], N-offset[s2]);
      if (errors>20) {
        printf("abort due to errors\n");
        return errors;
      }
    }
  }
  printf("finished with %d errors\n", errors);
  return errors;
}
