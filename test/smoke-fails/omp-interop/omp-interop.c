#include <omp.h>
#include <stdio.h>

int main(int argc, char **argv) {
  int EC;
  omp_interop_t IObj = omp_interop_none;

#pragma omp interop init(targetsync : IObj)

  const char *Type = omp_get_interop_str(IObj, omp_ipr_fr_id, &EC);
  if (EC) {
    printf("omp_get_interop_str failed: %d\n", EC);
    return -1;
  }

  printf("Interop Type: %s\n", Type);

  const char *Vendor = omp_get_interop_str(IObj, omp_ipr_vendor_name, &EC);
  if (EC) {
    printf("omp_get_interop_str failed: %d\n", EC);
    return -1;
  }

  printf("Interop Vendor ID: %s\n", Vendor);

  const char *Backend = omp_get_interop_str(IObj, omp_ipr_fr_name, &EC);
  if (EC) {
    printf("omp_get_interop_str failed: %d\n", EC);
    return -1;
  }

  printf("Interop Backend ID: %s\n", Backend);

#pragma omp interop destroy(IObj)

  return 0;
}

/// CHECK-NOT: failed
/// CHECK: Type: tasksync

/// CHECK-NOT: failed
/// CHECK: Vendor ID: amdhsa

/// CHECK-NOT: failed
/// CHECK: Backend ID: amdhsa backend
