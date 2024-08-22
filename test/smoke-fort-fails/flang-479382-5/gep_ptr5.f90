module array_target_enter
  implicit none
  contains
    subroutine test_array_target_parallel_do(dims)
      integer, intent(in) :: dims(2)
      double precision :: U(dims(1), dims(2))
      integer :: ix1, ix2
      !$omp target teams distribute parallel do simd collapse(2)
      do ix2 = 1, dims(2)
        do ix1 = 1, dims(1)
          U(ix1, ix2) = 0.0
        end do
      end do
    end subroutine test_array_target_parallel_do
end module array_target_enter
