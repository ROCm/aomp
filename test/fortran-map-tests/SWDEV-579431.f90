! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
  use iso_fortran_env, only : wp => real64
  implicit none
  type dtype_a
    real(wp), allocatable :: var_a(:,:,:) 
  end type
end module mod_a

module mod_b
  use mod_a
  implicit none
  type(dtype_a), allocatable :: var_b(:,:)
  integer :: var_c = 3, var_d = 4
  integer :: var_e = -4, var_f = 3
  integer :: var_g = 0, var_h = 0
  integer :: var_i = 0, var_j = 0

contains
  subroutine sub_a()
    integer :: var_k, var_l, var_m, var_n, var_o

    allocate(var_b(-1:1,1:var_c))
    !$omp target enter data map(alloc:var_b)
    
    do var_l = 1, var_c
      allocate(var_b(-1,var_l)%var_a(var_e:var_f,var_g:var_h,var_i:var_j))
      !$omp target enter data map(alloc:var_b(-1,var_l)%var_a)
    end do
    
    !-- Force host to have value
    do var_l = 1, var_c
      var_b(-1,var_l)%var_a = -var_l*100.0
    end do
    
    print*, 'Vals at init'
    do var_k = 1, var_c
      write(*,*) var_b(-1,var_k)%var_a
    end do
    
    !$omp target teams distribute parallel do collapse(4)
    do var_l = 1, var_c
      do var_o = var_i, var_j
        do var_n = var_g, var_h
          do var_m = var_e, var_f
            var_b(-1,var_l)%var_a(var_m,var_n,var_o) = 44
          end do
        end do
      end do
    end do
    
    print*
    print*, 'Vals after target loop (should be same as init)'
    do var_k = 1, var_c
      write(*,*) var_b(-1,var_k)%var_a

    do var_o = var_i, var_j
        do var_n = var_g, var_h
          do var_m = var_e, var_f
              if (var_b(-1,var_k)%var_a(var_m,var_n,var_o) /= -var_k*100.0) then
                print*, "======= FORTRAN Test Failed! ======="
                stop 1
              endif
          end do
        end do
      end do
    end do

    print*
    print*, 'Vals after update from device (should be mult. of Pi)'
    do var_k = 1, var_c
      !$omp target update from(var_b(-1,var_k)%var_a)
      write(*,*) var_b(-1,var_k)%var_a
        do var_o = var_i, var_j
            do var_n = var_g, var_h
                do var_m = var_e, var_f
                    if (var_b(-1,var_k)%var_a(var_m,var_n,var_o) /= 44) then
                        print*, "======= FORTRAN Test Failed! ======="
                        stop 1
                    endif
                end do
            end do
        end do
    end do

    print*, "======= FORTRAN Test passed! ======="
  end subroutine
end module

program prog_a
  use mod_b
  use iso_fortran_env
  implicit none
  print*, compiler_version()
  call sub_a()
end program
