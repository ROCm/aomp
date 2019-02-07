#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

// enable tests
#define CHECK              1
#define DEBUG              0

#define N (992)
#define INIT() INIT_LOOP(N, {A[i] = 0; C[i] = 1; D[i] = i; E[i] = -i;})

int main(void){
  #if CHECK
    check_offloading();
  #endif

  /*
   * Default device
   */
  printf("Is%s initial device\n", omp_is_initial_device() ? "" : " not");
  printf("Initial device: %d\n", omp_get_initial_device());
  omp_set_default_device(1);
  printf("Default device before task: %d\n", omp_get_default_device());
  #pragma omp task
  {
    printf("Default device inside task: %d\n", omp_get_default_device());
    omp_set_default_device(2);
    printf("Default device inside task after resetting: %d\n",
        omp_get_default_device());
  }
  #pragma omp taskwait
  printf("Default device outside task: %d\n", omp_get_default_device());

  // default device can set to whatever, if target fails, it goes to the host
  const int default_device = 0;
  omp_set_default_device(default_device);

  // default device for omp target call MUST be >= 0 and <omp_get_num_devices() or
  // the initial device. So when there are no devices, it must be the initial device
  int default_device_omp_target_call = default_device;
  if (omp_get_num_devices() == 0) {
    default_device_omp_target_call = omp_get_initial_device();
  } 
  #if DEBUG
    printf("test on machine with %d devices\n", omp_get_num_devices());
  #endif
  /*
   * Target alloc & target memcpy
   */
  double A[N], B[N], C[N], D[N], E[N];
  double *pA, *pB, *pC, *pD, *pE;
  // map ptrs
  pA = &A[0];
  pB = &B[0];
  pC = &C[0];
  pD = &D[0];
  pE = &E[0];

  INIT();

  pA = pA - 10;
  pC = pC - 20;
  pD = pD - 30;
  void *device_A = omp_target_alloc(N*sizeof(double), default_device_omp_target_call);
  void *device_C = omp_target_alloc(N*sizeof(double), default_device_omp_target_call);
  void *device_D = omp_target_alloc(N*sizeof(double), default_device_omp_target_call);
  double *dpA = (double *) device_A - 100;
  double *dpC = (double *) device_C - 200;
  double *dpD = (double *) device_D - 300;
  printf("omp_target_alloc %s\n", device_A && device_C && device_D ?
      "succeeded" : "failed");

  omp_target_memcpy(dpC, pC, N*sizeof(double), 200*sizeof(double),
      20*sizeof(double), default_device_omp_target_call, omp_get_initial_device());
  omp_target_memcpy(dpD, pD, N*sizeof(double), 300*sizeof(double),
      30*sizeof(double), default_device_omp_target_call, omp_get_initial_device());

  #pragma omp target is_device_ptr(dpA, dpC, dpD) device(default_device)
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      dpA[i+100] = dpC[i+200] + dpD[i+300] + 1;
  }

  omp_target_memcpy(pA, dpA, N*sizeof(double), 10*sizeof(double),
      100*sizeof(double), omp_get_initial_device(), default_device_omp_target_call);

  int fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test omp_target_memcpy: Failed\n");
  } else {
    printf ("Test omp_target_memcpy: Succeeded\n");
  }

  /*
   * target_is_present and target_associate/disassociate_ptr
   */
  INIT();
  if (offloading_disabled()) {
    // If offloading is disabled just recreate the messages so that this can
    // also be tested with no device.
    printf("C is not present, associating it...\n");
    printf("omp_target_associate_ptr C %s\n", 1 ? "succeeded" : "failed");
  } else if (!omp_target_is_present(C, default_device_omp_target_call)) {
    printf("C is not present, associating it...\n");
    int rc = omp_target_associate_ptr(C, dpC, N*sizeof(double),
        200*sizeof(double), default_device_omp_target_call);
    printf("omp_target_associate_ptr C %s\n", !rc ? "succeeded" : "failed");
  }
  if (offloading_disabled()) {
    // If offloading is disabled just recreate the messages so that this can
    // also be tested with no device.
    printf("D is not present, associating it...\n");
    printf("omp_target_associate_ptr D %s\n", 1 ? "succeeded" : "failed");
  } else if (!omp_target_is_present(D, default_device_omp_target_call)) {
    printf("D is not present, associating it...\n");
    int rc = omp_target_associate_ptr(D, dpD, N*sizeof(double),
        300*sizeof(double), default_device_omp_target_call);
    printf("omp_target_associate_ptr D %s\n", !rc ? "succeeded" : "failed");
  }
  #pragma omp target data map(from: C, D) device(default_device)
  {
    printf("Inside target data: A is%s present\n",
        (omp_target_is_present(A, default_device_omp_target_call) && !offloading_disabled()) ? "" : " not");
    printf("Inside target data: C is%s present\n",
        omp_target_is_present(C, default_device_omp_target_call) ? "" : " not");
    printf("Inside target data: D is%s present\n",
        omp_target_is_present(D, default_device_omp_target_call) ? "" : " not");

    // C and D are mapped "from", so there is no copy from host to device.
    // If the association was successful, their corresponding device arrays
    // are already populated from previous omp_target_memcpy with the correct
    // values and the following target for-loop must yield the correct results.
    #pragma omp target map(from: A) device(default_device)
    {
      #pragma omp parallel for schedule(static,1)
      for (int i = 0; i < 992; i++)
        A[i] = C[i] + D[i] + 1;
    }
  }


  if (offloading_disabled()) {
    printf("C is present, disassociating it...\n");
    printf("omp_target_disassociate_ptr C %s\n", 1 ? "succeeded" : "failed");
  } else if (omp_target_is_present(C, default_device_omp_target_call)) {
    printf("C is present, disassociating it...\n");
    int rc = omp_target_disassociate_ptr(C, default_device_omp_target_call);
    printf("omp_target_disassociate_ptr C %s\n", !rc ? "succeeded" : "failed");
  }
  if (offloading_disabled()) {
    printf("D is present, disassociating it...\n");
    printf("omp_target_disassociate_ptr D %s\n", 1 ? "succeeded" : "failed");
  } else if (omp_target_is_present(D, default_device_omp_target_call)) {
    printf("D is present, disassociating it...\n");
    int rc = omp_target_disassociate_ptr(D, default_device_omp_target_call);
    printf("omp_target_disassociate_ptr D %s\n", !rc ? "succeeded" : "failed");
  }

  fail = 0;
  VERIFY(0, N, A[i], (double)(i+2));
  if (fail) {
    printf ("Test omp_target_associate_ptr: Failed\n");
  } else {
    printf ("Test omp_target_associate_ptr: Succeeded\n");
  }

  omp_target_free(device_A, default_device_omp_target_call);
  omp_target_free(device_C, default_device_omp_target_call);
  omp_target_free(device_D, default_device_omp_target_call);

  return 0;
}
