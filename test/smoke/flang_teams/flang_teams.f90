program TargetReduction_3D

    implicit none
    
    integer, dimension ( 3,3,3 ) ::  iV
    integer i,j,k
    
!   !$OMP  target enter data map ( from : iV)

     print*,'Starting'
    !$OMP  target MAP(from:iV)
    !$OMP  teams
    !$OMP  distribute private (i)
    do i = 1, 3
      !$omp parallel  private ( j, k )
      do j = 1, 3 
        !$omp do
        do k = 1, 3
            iV(i,j,k) = i+j+k
        end do
        !$omp end do
      end do
      !$omp end parallel
    end do
    !$OMP  end distribute
    !$OMP END TEAMS
    !$OMP END TARGET
    print*, 'MaxSpeed'
    print*, iV
end program TargetReduction_3D

