module kernels
    interface kernel
        module procedure :: kernel_1d, kernel_2d, kernel_3d, &
                            kernel_4d, kernel_5d, kernel_6d, &
                            kernel_7d
    end interface
contains
    subroutine kernel_1d(array)
        implicit none
        real, dimension(:) :: array
        integer :: i

        print *, 'kernel_1d'
        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do
        do i=1, ubound(array, 1)
            array(i) = 42.0
        end do
        print *, 'kernel_1d end'
    end subroutine

    subroutine kernel_2d(array)
        implicit none
        real, dimension(:,:) :: array
        integer :: i, j

        print *, 'kernel_2d'
        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do collapse(2)
        do j=1, ubound(array, 2)
            do i=1, ubound(array, 1)
                array(i,j) = 42.0
            end do
        end do
        print *, 'kernel_2d end'
    end subroutine

    subroutine kernel_3d(array)
        implicit none
        real, dimension(:,:,:) :: array
        integer :: i, j, k

        print *, 'kernel_3d'
        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do collapse(3)
        do k=1, ubound(array, 3)
           do j=1, ubound(array, 2)
               do i=1, ubound(array, 1)
                   array(i,j,k) = 42.0
               end do
           end do
        end do
        print *, 'kernel_3d end'
    end subroutine

    subroutine kernel_4d(array)
        implicit none
        real, dimension(:,:,:,:) :: array
        integer :: i, j, k, l

        print *, 'kernel_4d'
        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do collapse(4)
        do l=1, ubound(array, 4)
           do k=1, ubound(array, 3)
              do j=1, ubound(array, 2)
                  do i=1, ubound(array, 1)
                      array(i,j,k,l) = 42.0
                  end do
              end do
           end do
        enddo
        print *, 'kernel_4d end'
    end subroutine

    subroutine kernel_5d(array)
        implicit none
        real, dimension(:,:,:,:,:) :: array
        integer :: i, j, k, l, m

        print *, 'kernel_5d'
        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do collapse(5)
        do m=1, ubound(array, 5)
           do l=1, ubound(array, 4)
              do k=1, ubound(array, 3)
                 do j=1, ubound(array, 2)
                     do i=1, ubound(array, 1)
                         array(i,j,k,l,m) = 42.0
                     end do
                 end do
              end do
           enddo
        enddo
        print *, 'kernel_5d end'
    end subroutine

    subroutine kernel_6d(array)
        implicit none
        real, dimension(:,:,:,:,:,:) :: array
        integer :: i, j, k, l, m, n

        print *, 'kernel_6d'
        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do collapse(6)
        do n=1, ubound(array, 6)
           do m=1, ubound(array, 5)
              do l=1, ubound(array, 4)
                 do k=1, ubound(array, 3)
                    do j=1, ubound(array, 2)
                        do i=1, ubound(array, 1)
                            array(i,j,k,l,m,n) = 42.0
                        end do
                    end do
                 end do
              enddo
           enddo
        enddo
        print *, 'kernel_6d end'
    end subroutine

    subroutine kernel_7d(array)
        implicit none
        real, dimension(:,:,:,:,:,:,:) :: array
        integer :: i, j, k, l, m, n, o

        print *, 'kernel_7d'
        !$omp target enter data map(alloc:array)
        !$omp target teams distribute parallel do collapse(7)
        do o=1, ubound(array, 7)
           do n=1, ubound(array, 6)
              do m=1, ubound(array, 5)
                 do l=1, ubound(array, 4)
                    do k=1, ubound(array, 3)
                       do j=1, ubound(array, 2)
                           do i=1, ubound(array, 1)
                               array(i,j,k,l,m,n,o) = 42.0
                           end do
                       end do
                    end do
                 enddo
              enddo
           enddo
        enddo
        print *, 'kernel_7d end'
    end subroutine
end module

module wrap
    use kernels, only: kernel
end module

program map_alloc
    use wrap, only: kernel
    implicit none

    integer, parameter :: n = 10
    real :: array1(n            )
    real :: array2(n,n          )
    real :: array3(n,n,n        )
    real :: array4(n,n,n,n      )
    real :: array5(n,n,n,n,n    )
    real :: array6(n,n,n,n,n,n  )
    real :: array7(n,n,n,n,n,n,n)

    call kernel(array1)

    call kernel(array2)

    call kernel(array3)

    call kernel(array4)

    call kernel(array5)

    call kernel(array6)

    call kernel(array7)
end program