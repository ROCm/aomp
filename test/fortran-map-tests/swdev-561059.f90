! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
  implicit none
contains
  subroutine sub_a(var_a, var_b, var_c, var_d)
    implicit none
    integer, intent(in) :: var_a
    real(8), intent(in) :: var_b(var_a)
    real(8), intent(in) :: var_c(var_a)
    real(8), intent(out) :: var_d(var_a)
    integer :: var_e
    !$omp target data map(present, alloc:var_b, var_c, var_d)
    !$omp end target data

    !$omp target teams distribute parallel do
    do var_e=1,var_a
       var_d(var_e) = var_c(var_e) + var_b(var_e)
    end do

  end subroutine sub_a
end module mod_a

program prog_a
  use mod_a, only : sub_a
  implicit none
  integer :: var_a, var_b, var_c, var_d, var_e, var_f, var_g, var_h, var_i, var_j, var_k
  real(kind=8), allocatable :: var_l(:,:,:)
  real(kind=8), allocatable :: var_m(:,:,:)
  CHARACTER(LEN=256) :: var_n
  INTEGER :: var_o
  INTEGER :: var_p

  CALL GET_COMMAND_ARGUMENT(1, var_n, STATUS=var_p)
  READ(var_n, *, IOSTAT=var_p) var_a
  CALL GET_COMMAND_ARGUMENT(2, var_n, STATUS=var_p)
  READ(var_n, *, IOSTAT=var_p) var_b
  CALL GET_COMMAND_ARGUMENT(3, var_n, STATUS=var_p)
  READ(var_n, *, IOSTAT=var_p) var_c

  CALL GET_COMMAND_ARGUMENT(4, var_n, STATUS=var_p)
  READ(var_n, *, IOSTAT=var_p) var_g
  CALL GET_COMMAND_ARGUMENT(5, var_n, STATUS=var_p)
  READ(var_n, *, IOSTAT=var_p) var_h
  CALL GET_COMMAND_ARGUMENT(6, var_n, STATUS=var_p)
  READ(var_n, *, IOSTAT=var_p) var_j
  CALL GET_COMMAND_ARGUMENT(7, var_n, STATUS=var_p)
  READ(var_n, *, IOSTAT=var_p) var_i
  CALL GET_COMMAND_ARGUMENT(8, var_n, STATUS=var_p)
  READ(var_n, *, IOSTAT=var_p) var_k

  allocate(var_l(1:var_a,1:var_b,1:var_c))
  allocate(var_m(1:var_a,1:var_b,1:var_c))

  do var_d=1, var_a
     do var_e=1, var_b
        do var_f=1, var_c
           var_l(var_d,var_e,var_f) = var_e
        end do
     end do
  end do

  do var_f=1, var_c
     do var_d=1,var_a
        var_m(var_d,var_i,var_f) = var_l(var_d,var_g,var_f) + var_l(var_d,var_k+1,var_f)
        var_m(var_d,var_k,var_f) = var_l(var_d,var_h,var_f) + var_l(var_d,var_j,var_f)
     end do

     !$OMP TARGET ENTER DATA MAP(ALLOC:var_l(1:var_a,:,var_f))

     !$OMP TARGET UPDATE TO(var_l(1:var_a,var_g:var_h,var_f), &
     !$OMP&                 var_l(1:var_a,var_k+1:var_j,var_f))

     call sub_a(var_a,var_l(:,var_g,var_f),var_l(:,var_k+1,var_f),var_l(:,var_i,var_f))
     call sub_a(var_a,var_l(:,var_h,var_f),var_l(:,var_j,var_f),var_l(:,var_k,var_f))

     !$OMP TARGET UPDATE FROM(var_l(1:var_a,var_i:var_k,var_f))
     !$OMP TARGET EXIT DATA MAP(DELETE:var_l(1:var_a,:,var_f))

    if (var_m(1,var_k,var_f) /= var_l(1,var_k,var_f)) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    endif

    if (var_m(2,var_i,var_f) /= var_l(2,var_i,var_f)) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    endif

    if (var_m(var_a,var_i,var_f) /= var_l(var_a,var_i,var_f)) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    endif

    if (var_m(1,var_k,var_f) /= var_l(1,var_k,var_f)) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    endif

    if (var_m(2,var_k,var_f) /= var_l(2,var_k,var_f)) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    endif

    if (var_m(var_a,var_k,var_f) /= var_l(var_a,var_k,var_f)) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    endif
  end do


  deallocate(var_l)

  print *, "======= FORTRAN Test Passed! ======="
end program prog_a
