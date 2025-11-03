module foo
  implicit none
contains
  subroutine bar(n, a, b, c)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: a(n)
    real(8), intent(in) :: b(n)
    real(8), intent(out) :: c(n)
    integer :: i
    !$omp target data map(present, alloc:a, b, c)
    !$omp end target data

    !$omp target teams distribute parallel do
    do i=1,n
       c(i) = b(i) + a(i)
    end do

  end subroutine bar
end module foo

program main
  use foo, only : bar
  implicit none
  integer :: n1, n2, n3, i1, i2, i3, iinbeg, iinend, ioutbeg, ioutend, ifldstot, choice
  real(kind=8), allocatable :: zrgp(:,:,:) ! monolithic IFS data structure
  real(kind=8), allocatable :: cpu(:,:,:) ! monolithic IFS data structure
  CHARACTER(LEN=256) :: arg_string
  INTEGER :: my_integer_value
  INTEGER :: status

  CALL GET_COMMAND_ARGUMENT(1, arg_string, STATUS=status)
  READ(arg_string, *, IOSTAT=status) choice

  IF (choice == 0) THEN
     ! This size works.
     n1=100
     n2=100
     n3=1
  ELSE IF (choice == 1) THEN
     ! This size fails
     n1=100
     n2=101
     n3=1
  ELSE
     ! This choice creates a a failure with present alloc similar to what's seen in ECRad.
     n2=100
     n3=2
     n1=50
  END IF

  iinbeg= 1
  iinend= 30
  ifldstot= n2
  ioutbeg = iinend+1
  ioutend = 60

  allocate(zrgp(1:n1,1:n2,1:n3))
  allocate(cpu(1:n1,1:n2,1:n3))

  do i1=1, n1
     do i2=1, n2
        do i3=1, n3
           zrgp(i1,i2,i3) = i2
        end do
     end do
  end do

  do i3=1, n3
     do i1=1,n1
        cpu(i1,ioutbeg,i3) = zrgp(i1,iinbeg,i3) + zrgp(i1,ioutend+1,i3)
        cpu(i1,ioutend,i3) = zrgp(i1,iinend,i3) + zrgp(i1,ifldstot,i3)
     end do

     !$OMP TARGET ENTER DATA MAP(ALLOC:zrgp(1:n1,:,i3))

     !$OMP TARGET UPDATE TO(zrgp(1:n1,iinbeg:iinend,i3), &
     !$OMP&                 zrgp(1:n1,ioutend+1:ifldstot,i3))

     call bar(n1,zrgp(:,iinbeg,i3),zrgp(:,ioutend+1,i3),zrgp(:,ioutbeg,i3))
     call bar(n1,zrgp(:,iinend,i3),zrgp(:,ifldstot,i3),zrgp(:,ioutend,i3))

     !$OMP TARGET UPDATE FROM(zrgp(1:n1,ioutbeg:ioutend,i3))
     !$OMP TARGET EXIT DATA MAP(DELETE:zrgp(1:n1,:,i3))

     write(*,*) "CPU(:,",ioutbeg,",",i3,")=", cpu(1,ioutbeg,i3), cpu(2,ioutbeg,i3), cpu(n1,ioutbeg,i3)
     write(*,*) "GPU(:,",ioutbeg,",",i3,")=", zrgp(1,ioutbeg,i3), zrgp(2,ioutbeg,i3), zrgp(n1,ioutbeg,i3)

     write(*,*) "CPU(:,",ioutend,",",i3,")=", cpu(1,ioutend,i3), cpu(2,ioutend,i3), cpu(n1,ioutend,i3)
     write(*,*) "GPU(:,",ioutend,",",i3,")=", zrgp(1,ioutend,i3), zrgp(2,ioutend,i3), zrgp(n1,ioutend,i3)

    if (cpu(1,ioutend,i3) /= zrgp(1,ioutend,i3) .OR. cpu(2,ioutbeg,i3) /= zrgp(2,ioutbeg,i3) .OR. &
        cpu(n1,ioutbeg,i3) /= zrgp(n1,ioutbeg,i3) .OR. cpu(1,ioutend,i3) /= zrgp(1,ioutend,i3) .OR. &
        cpu(2,ioutend,i3) /= zrgp(2,ioutend,i3) .OR. cpu(n1,ioutend,i3) /= zrgp(n1,ioutend,i3)) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    endif
  end do

  deallocate(zrgp)

  ! Reached end without early exiting above, results were correct.
  print *, "======= FORTRAN Test PASSED! ======="
end program main