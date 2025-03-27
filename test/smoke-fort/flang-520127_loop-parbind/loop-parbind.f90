!************modified reproducer with parallel**********
PROGRAM rep_loopbind

      implicit none

      integer :: i,j
      real(kind=8) :: tmp


      !$omp target teams loop bind(teams)
      do j=1,1000
        !$omp parallel
        !$omp loop bind(parallel) private(tmp)
        do i=1,512

        end do
        !$omp end loop
        !$omp end parallel
      end do

END PROGRAM
!*****************
