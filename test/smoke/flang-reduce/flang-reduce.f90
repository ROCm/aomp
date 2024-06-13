program main
    real(8) ::x(8192)
    real(8) ::y(8192)
    real(8) ::cgdot
    integer :: i,N
    N = 8192
    do i=1,N
      x(i) = i * 0.5
      y(i) = i * 0.5
    enddo
    cgdot = 0.0
    !$omp target teams distribute parallel do reduction(+:cgdot)
    do i=1,N
       cgdot = cgdot+x(i)+y(i)
    enddo
end program main
