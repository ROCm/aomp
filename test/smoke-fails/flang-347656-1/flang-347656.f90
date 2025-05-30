module mpas_dmpar
  real, dimension(:,:), allocatable :: globalarray5

  contains

  subroutine alloc_global(M,N,O,P)
    integer :: M,N,O,P
    allocate(globalarray5(M,P))
    !$omp target enter data map(alloc: globalarray5)
  end subroutine alloc_global
  
end module mpas_dmpar

module mpas_physics

  contains

  subroutine fill_5(globalarray5,M,N,P)
    integer :: M,N,P,i,j
    !real, dimension(M,P)   :: globalarray5 ! workaround
    real, dimension(M,N+1) :: globalarray5
    !$omp target teams distribute parallel do simd collapse(2)
    do i=1,P
      do j=1,M
        globalarray5(j,i) = 0.0
      end do
    end do
    !$omp end target teams distribute parallel do simd
  end subroutine fill_5

end module mpas_physics

program mpas_driver
  use mpas_dmpar
  use mpas_physics
  call alloc_global(26,21176,63873,20480)
  call fill_5(globalarray5,26,21176,20480)

end program mpas_driver
