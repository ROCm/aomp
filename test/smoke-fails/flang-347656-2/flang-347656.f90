module mpas_dmpar
  real,dimension(:),        pointer :: globalarray6

  contains

  subroutine alloc_global(M,N,O,P)
    integer :: M,N,O,P
    allocate(globalarray6(O+1))
    !$omp target enter data map(alloc: globalarray6)
  end subroutine alloc_global
  
end module mpas_dmpar

module mpas_physics

  contains

  subroutine fill_6(globalarray6,O)
    integer :: O,i
    !real, dimension(O+1)          :: globalarray6 ! workaround
    real, dimension(:), pointer   :: globalarray6
    !$omp target teams distribute parallel do simd
    do i=1,O
      globalarray6(i) = 0.0
    end do
    !$omp end target teams distribute parallel do simd

  end subroutine fill_6
end module mpas_physics

program mpas_driver
  use mpas_dmpar
  use mpas_physics
  call alloc_global(26,21176,63873,20480)
  call fill_6(globalarray6,63873)

end program mpas_driver
