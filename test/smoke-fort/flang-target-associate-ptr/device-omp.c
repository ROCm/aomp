#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void *AllocateTargetDouble_OMP(int nValues) {
  int iDevice;
  void *D_Pointer;

  D_Pointer = NULL;
  iDevice = omp_get_default_device();

  D_Pointer = omp_target_alloc(sizeof(double) * nValues, iDevice);

  return D_Pointer;
}

int AssociateTargetDouble_OMP(void *Host, void *Device, int nValues,
                              int oValue) {
  int iDevice, retval;
  size_t Size, Offset;

  retval = -1;

  Size = sizeof(double) * nValues;
  Offset = sizeof(double) * oValue;
  iDevice = omp_get_default_device();

  retval = omp_target_associate_ptr(Host, Device, Size, Offset, iDevice);

  return retval;
}

void FreeTarget_OMP(void *D_Pointer) {
  int iDevice;

  iDevice = omp_get_default_device();
  omp_target_free(D_Pointer, iDevice);
}

int DisassociateTarget_OMP(void *Host) {
  int iDevice, retval;

  retval = -1;

  iDevice = omp_get_default_device();

  retval = omp_target_disassociate_ptr(Host, iDevice);

  return retval;
}

