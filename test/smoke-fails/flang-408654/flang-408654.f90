        module aux
         implicit none

         private 

         public :: chi_summator_t, matmult
         public :: this

         type chi_summator_t
            double precision :: fact
            double complex, pointer :: gmetempr(:,:),gmetempc(:,:),chilocal(:,:)
            double complex, allocatable :: other(:,:)
            double complex, pointer :: gme(:,:,:,:,:,:)
            integer :: stuff(20)
         end type chi_summator_t

         type(chi_summator_t) :: this
        contains 

        subroutine matmult( A, B, C, iii, jjj, kkk )

          double complex, intent(in) :: A(:,:), B(:,:)
          double complex, intent(out) :: C(:,:)
          integer, intent(in) :: iii, jjj, kkk

          integer :: i, j, k

          ! !$omp target data use_device_ptr(a, b, c)
          !$OMP target teams distribute parallel do thread_limit(512) collapse ( 2 )
          do i = 1, iii
            do j = 1, jjj
              C(i,j) = (0.0D+00,0.0D+00)
              do k = 1, kkk
                C(i,j) = C(i,j) + A(i,k) * B(k,j)
              end do
            end do
          end do
          ! !$omp end target data

        end subroutine matmult

        end module aux 

        program test
        use aux
        implicit none
        ! type(chi_summator_t) :: this
        integer :: iii, jjj, kkk

        iii = 1000
        jjj = 500
        kkk = 2000

        this%fact = 1.0D+00
        allocate(this%chilocal(iii,jjj))
        allocate(this%gmetempr(iii,kkk))
        allocate(this%gmetempc(kkk,jjj))
        allocate(this%gme(kkk,jjj,iii,kkk,jjj,iii))

        this%chilocal = (1.0,2.0)
        this%gmetempr = (2.0,1.0)
        this%gmetempc = (3.0,0.0)
        this%gme = (5.0,55.0)

        do iii = 1, 10
          ! !$omp target enter data map (to: this)
          !$omp target enter data map (to: this%chilocal, this%gmetempr, this%gmetempc)

          call matmult( this%gmetempr, this%gmetempc, this%chilocal, iii, jjj, kkk)
          !$OMP target update from ( this%chilocal )

          !$omp target exit data map(delete: this%chilocal, this%gmetempr, this%gmetempc)
          ! !$omp target exit data map(delete: this)
        end do

        deallocate(this%chilocal, this%gmetempr, this%gmetempc)

        end program 
