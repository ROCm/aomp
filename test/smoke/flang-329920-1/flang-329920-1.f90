module mpas_dmpar
  real, dimension(:,:), allocatable :: globalarray1,globalarray4,globalarray2,globalarray3,globalarray5

  contains

  subroutine alloc_global(M,N,O)
    integer :: M,N,O
    allocate(globalarray1(M,N+1))
    allocate(globalarray2(M,N+1))
    allocate(globalarray3(M,N+1))
    allocate(globalarray4(M,O+1))
    allocate(globalarray5(M,N-1))
    !$omp target enter data map(alloc:globalarray1,globalarray2,globalarray3,globalarray4,globalarray5)
  end subroutine alloc_global

  subroutine fill_global(M,N)
    integer :: N,i,j
    !$omp target teams distribute parallel do simd collapse(2)
    do i=1,N
      do j=1,M
        globalarray1(j,i) = 0.0
      end do
    end do
    !$omp end target teams distribute parallel do simd
  end subroutine fill_global
  
  subroutine fill_global2(M,N,O)
    integer :: M,N,O,i,j
    !$omp target teams distribute parallel do simd collapse(2)
    do i=1,O
      do j=1,M
        globalarray4(j,i) = 0.0
      end do
    end do
    !$omp end target teams distribute parallel do simd
  end subroutine fill_global2
  
end module mpas_dmpar

program mpas_driver
  use mpas_dmpar
  call alloc_global(26,21176,63874)
  call fill_global2(26,21176,63874)
  print *, "PASS"
  return
end program mpas_driver
