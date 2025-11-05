subroutine generic_parallel_reduction(output_var)
  implicit none
  integer, intent(out) :: output_var

  integer :: i
  integer :: private_var

  !$omp target private(private_var) map(from: output_var)
    private_var = 0

    !$omp parallel do reduction(+:private_var)
    do i = 1, 32
      private_var = private_var + 1
    end do

    output_var = private_var
  !$omp end target
end subroutine

program main
  implicit none
  integer :: val

  call generic_parallel_reduction(val)
  if ( val .ne. 32 ) then
    print *, 'Unexpected result:', val
    stop 1
  end if
end program
