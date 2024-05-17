module types
    integer, parameter :: bSize = 128
    type field2DReal

       ! Raw array holding field data on this block
       real, dimension(:,:), pointer :: array => null()

       ! Pointers to the prev and next blocks for this field on this task
       type (field2DReal), pointer :: prev => null()
       type (field2DReal), pointer :: next => null()

    end type field2DReal

    contains 
    subroutine init_ptr_2D(ptr)
         type (field2DReal), pointer , intent(inout):: ptr
         integer :: i

         allocate(ptr)
         allocate(ptr%array(bSize,bSize))
         do i=1,128
             ptr%array(i,i) = i
         end do

         !$omp target enter data map(to:ptr%array)
    end subroutine init_ptr_2D
end module types

program test
    use types

    type (field2DReal), pointer :: head
    type (field2DReal), pointer :: tail
    real, dimension(:,:), pointer :: ary
    integer :: i

    call init_ptr_2D(head)
    call init_ptr_2D(head%next)
    tail => head%next

    ary => head%array
!$omp target teams distribute
    do i=1,bSize
        ary(i,i) = ary(i,i) * 2
    end do

    ary => tail%array
!$omp target teams distribute 
    do i=1,bSize
        ary(i,i) = ary(i,i) * 2
    end do

!$omp target exit data map(from:head%array)
!$omp target exit data map(from:tail%array)

    do i=1,10
        print *, head%array(i,i)
    end do    
    do i=1,10
        print *, tail%array(i,i)
    end do    
    do i=1,10
       if ( head%array(i,i) .ne. tail%array(i,i)) stop 2
    end do
    print *, "PASS"
    return
end program test
