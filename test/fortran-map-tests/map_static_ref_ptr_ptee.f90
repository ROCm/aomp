! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    interface sub_a
        module procedure :: sub_a_1d, sub_a_2d, sub_a_3d, &
                            sub_a_4d, sub_a_5d, sub_a_6d, &
                            sub_a_7d
    end interface
contains
    subroutine sub_a_1d(var_a)
        implicit none
        real, dimension(:) :: var_a
        integer :: var_b

        print *, 'sub_a_1d'
        !$omp target enter data map(ref_ptee, storage:var_a)
        !$omp target teams distribute parallel do
        do var_b=1, ubound(var_a, 1)
            var_a(var_b) = 42.0
        end do
        print *, 'sub_a_1d end'
    end subroutine

    subroutine sub_a_2d(var_a)
        implicit none
        real, dimension(:,:) :: var_a
        integer :: var_b, var_c

        print *, 'sub_a_2d'
        !$omp target enter data map(ref_ptee, storage:var_a)
        !$omp target teams distribute parallel do collapse(2)
        do var_c=1, ubound(var_a, 2)
            do var_b=1, ubound(var_a, 1)
                var_a(var_b,var_c) = 42.0
            end do
        end do
        print *, 'sub_a_2d end'
    end subroutine

    subroutine sub_a_3d(var_a)
        implicit none
        real, dimension(:,:,:) :: var_a
        integer :: var_b, var_c, var_d

        print *, 'sub_a_3d'
        !$omp target enter data map(ref_ptee, storage:var_a)
        !$omp target teams distribute parallel do collapse(3)
        do var_d=1, ubound(var_a, 3)
           do var_c=1, ubound(var_a, 2)
               do var_b=1, ubound(var_a, 1)
                   var_a(var_b,var_c,var_d) = 42.0
               end do
           end do
        end do
        print *, 'sub_a_3d end'
    end subroutine

    subroutine sub_a_4d(var_a)
        implicit none
        real, dimension(:,:,:,:) :: var_a
        integer :: var_b, var_c, var_d, var_e

        print *, 'sub_a_4d'
        !$omp target enter data map(ref_ptee, storage:var_a)
        !$omp target teams distribute parallel do collapse(4)
        do var_e=1, ubound(var_a, 4)
           do var_d=1, ubound(var_a, 3)
              do var_c=1, ubound(var_a, 2)
                  do var_b=1, ubound(var_a, 1)
                      var_a(var_b,var_c,var_d,var_e) = 42.0
                  end do
              end do
           end do
        enddo
        print *, 'sub_a_4d end'
    end subroutine

    subroutine sub_a_5d(var_a)
        implicit none
        real, dimension(:,:,:,:,:) :: var_a
        integer :: var_b, var_c, var_d, var_e, var_f

        print *, 'sub_a_5d'
        !$omp target enter data map(ref_ptee, storage:var_a)
        !$omp target teams distribute parallel do collapse(5)
        do var_f=1, ubound(var_a, 5)
           do var_e=1, ubound(var_a, 4)
              do var_d=1, ubound(var_a, 3)
                 do var_c=1, ubound(var_a, 2)
                     do var_b=1, ubound(var_a, 1)
                         var_a(var_b,var_c,var_d,var_e,var_f) = 42.0
                     end do
                 end do
              end do
           enddo
        enddo
        print *, 'sub_a_5d end'
    end subroutine

    subroutine sub_a_6d(var_a)
        implicit none
        real, dimension(:,:,:,:,:,:) :: var_a
        integer :: var_b, var_c, var_d, var_e, var_f, var_g

        print *, 'sub_a_6d'
        !$omp target enter data map(ref_ptee, storage:var_a)
        !$omp target teams distribute parallel do collapse(6)
        do var_g=1, ubound(var_a, 6)
           do var_f=1, ubound(var_a, 5)
              do var_e=1, ubound(var_a, 4)
                 do var_d=1, ubound(var_a, 3)
                    do var_c=1, ubound(var_a, 2)
                        do var_b=1, ubound(var_a, 1)
                            var_a(var_b,var_c,var_d,var_e,var_f,var_g) = 42.0
                        end do
                    end do
                 end do
              enddo
           enddo
        enddo
        print *, 'sub_a_6d end'
    end subroutine

    subroutine sub_a_7d(var_a)
        implicit none
        real, dimension(:,:,:,:,:,:,:) :: var_a
        integer :: var_b, var_c, var_d, var_e, var_f, var_g, var_h

        print *, 'sub_a_7d'
        !$omp target enter data map(ref_ptee, storage:var_a)
        !$omp target teams distribute parallel do collapse(7)
        do var_h=1, ubound(var_a, 7)
           do var_g=1, ubound(var_a, 6)
              do var_f=1, ubound(var_a, 5)
                 do var_e=1, ubound(var_a, 4)
                    do var_d=1, ubound(var_a, 3)
                       do var_c=1, ubound(var_a, 2)
                           do var_b=1, ubound(var_a, 1)
                               var_a(var_b,var_c,var_d,var_e,var_f,var_g,var_h) = 42.0
                           end do
                       end do
                    end do
                 enddo
              enddo
           enddo
        enddo
        print *, 'sub_a_7d end'
    end subroutine
end module

module mod_b
    use mod_a, only: sub_a
end module

program prog_a
    use mod_b, only: sub_a
    implicit none

    integer, parameter :: var_a = 10
    real :: var_b(var_a            )
    real :: var_c(var_a,var_a          )
    real :: var_d(var_a,var_a,var_a        )
    real :: var_e(var_a,var_a,var_a,var_a      )
    real :: var_f(var_a,var_a,var_a,var_a,var_a    )
    real :: var_g(var_a,var_a,var_a,var_a,var_a,var_a  )
    real :: var_h(var_a,var_a,var_a,var_a,var_a,var_a,var_a)

    call sub_a(var_b)

    call sub_a(var_c)

    call sub_a(var_d)

    call sub_a(var_e)

    call sub_a(var_f)

    call sub_a(var_g)

    call sub_a(var_h)

    print*, "======= FORTRAN Test Passed! ======="
end program
