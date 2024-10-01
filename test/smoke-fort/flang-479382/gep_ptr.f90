module array_target_enter
  implicit none

  contains
    subroutine test_array_target_enter_data(dims)
      integer, intent(in) :: dims(2)
      double precision :: U(2, dims(2))

      U = 0.0
      !$omp target enter data map(to: dims, U)
      U = 1.0
      !$omp target exit data map(release: U, dims)
    end subroutine test_array_target_enter_data
end module array_target_enter
