#include <omp.h>
#include <stdio.h>
int main() {
  #pragma omp parallel for
  for (int j = 0; j < 2; j++) {
    int host_thread= omp_get_thread_num();
    printf("%d - 1. On Host: CPU Parallel\n", host_thread);
  #pragma omp target map(host_thread)
  {
    int p = 20;
    printf("%d - 2. On Target: Begin Target\n", host_thread);
    #pragma omp parallel for
    for (int i = 0; i < 2; i++) {
      int d = 0;
        printf("%d - 3. On Target: First Parallel Section\n", host_thread);
    }
    printf("%d - 4. On Target: End First Parallel Section. Master Thread ID: %d\n", host_thread, omp_get_thread_num());
    #pragma omp parallel for
    for (int i = 0; i < 2; i++) {
      printf("%d - 5. On Target: Second Parallel Section\n", host_thread);
    }
    printf("%d - 6. On Target: End Second Parallel Section. Master Thread ID: %d\n", host_thread, omp_get_thread_num());
  }
    printf("%d - 7. On Host: End Target\n", host_thread);
  }
  return 0;
}
