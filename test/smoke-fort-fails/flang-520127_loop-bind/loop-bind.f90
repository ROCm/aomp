!********Example how it is used in the code************
PROGRAM rep_loopbind

      implicit none

      integer :: i,j
      real(kind=8) :: tmp


      !$omp target teams loop bind(teams)
      do j=1,1000
        !$omp loop bind(parallel) private(tmp)
        do i=1,512

        end do
      end do

END PROGRAM
!*********end example***********
