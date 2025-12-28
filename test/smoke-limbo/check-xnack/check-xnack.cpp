#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include <iostream>

void f() {
  bool hasSystemXnackEnabled = false;
  hsa_status_t HsaStatus = hsa_system_get_info(
      HSA_AMD_SYSTEM_INFO_XNACK_ENABLED, &hasSystemXnackEnabled);
  if (HsaStatus != HSA_STATUS_SUCCESS)
    printf("Output status is bad!\n");
  printf("hasSystemXnackEnabled = %d\n", hasSystemXnackEnabled);
}

int main() {
  // CHECK-NOT: Output status is bad!
  // CHECK: hasSystemXnackEnabled = 1
  f();
  return 0;
}
