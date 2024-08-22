module array_target_enter
  implicit none
  integer :: nX
  contains
    subroutine test_array_target_enter
      double precision :: dummy(2,nX)
      !$omp target enter data map(alloc: dummy)
    end subroutine test_array_target_enter
end module array_target_enter
