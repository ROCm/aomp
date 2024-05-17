program minimal
  use iso_c_binding

  implicit none

  TYPE descriptor_t
    INTEGER, ALLOCATABLE :: nsp(:)
  END TYPE

  INTEGER, ALLOCATABLE :: nsp2(:)


  TYPE(descriptor_t) :: desc

  ! Fails with
  !F90-F-0000-Internal compiler error. gen_sptr(): unexpected storage type       6  (unexpected_storage_type.f90: 23)
  !F90-F-0000-Internal compiler error. gen_sptr(): unexpected storage type       6  (unexpected_storage_type.f90: 23)
  ! Works
  !!$omp target update to(nsp2)
  !$omp target update to(desc%nsp)
  print *, "PASS"
  return

end program minimal
