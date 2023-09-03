program main
    real(8) ::x(2048)
    real(8) ::y(2048)
    real(8) ::cgdot
    integer :: i,N
    N = 2408
    do i=1,Niters
      x(i) = i * 0.5
      y(i) = i * 0.5
    enddo
    cgdot = 0.0
    !$omp target teams distribute parallel do reduction(+:cgdot)
    do i=1,N
       cgdot = cgdot+x(i)+y(i)
    enddo
end program main
