program test
    type field2DReal
       real, dimension(:), pointer :: array => null()
    end type field2DReal

    type (field2DReal) :: head
    real, dimension(:), pointer :: arr
    integer :: i, use_gpu = 1
    allocate(head%array(128));
    arr => head%array

!$omp target teams distribute if(use_gpu) 
    do i=1,128
        arr(i) = i
    end do

!$omp target exit data map(from:head%array)

    do i=1,128
        print *, head%array(i)
    end do

end program test
