module array_target_enter
  implicit none
  contains
    subroutine test_array_target_parallel_do(dims)
      integer, intent(in) :: dims(2)
      ! fails
      double precision :: U(2, dims(2))
      ! works
      ! double precision :: U(dims(1), dims(2))
      integer :: ix1, ix2
      !$omp target parallel do
      do ix2 = 1, 2
        do ix1 = 1, 2
          U(ix1, ix2) = 0.0
        end do
      end do
    end subroutine test_array_target_parallel_do
end module array_target_enter
