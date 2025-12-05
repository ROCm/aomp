PROGRAM DisableOffload
  USE OMP_LIB
  USE ISO_C_BINDING

  IMPLICIT NONE

  INTEGER, PARAMETER :: N = 10000000
  REAL, ALLOCATABLE, target :: host_array(:)
  TYPE(C_PTR) :: c_host_ptr
  INTEGER :: device_num
  LOGICAL :: is_present_on_device
  integer i


  ALLOCATE(host_array(N))
  host_array = 1.0

  call omp_set_default_device(2);


  device_num = OMP_GET_DEFAULT_DEVICE()


  c_host_ptr = C_LOC(host_array)


  is_present_on_device = OMP_TARGET_IS_PRESENT(c_host_ptr, device_num)

  print *, 'Data is on device ', is_present_on_device

  !$OMP TARGET ENTER DATA MAP(TO: host_array)

  is_present_on_device = OMP_TARGET_IS_PRESENT(c_host_ptr, device_num)

  print *, 'Data is on device ', is_present_on_device

  !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO  &
  !$OMP& map(present, alloc: host_array)
  DO i = 1, N
    host_array(i) = host_array(i) * 2.0
  END DO

  !$OMP TARGET EXIT DATA MAP(FROM: host_array)

  is_present_on_device = OMP_TARGET_IS_PRESENT(c_host_ptr, device_num)

  print *, 'Data is on device ', is_present_on_device

  print *, host_array(55)

  DEALLOCATE(host_array)

END PROGRAM
