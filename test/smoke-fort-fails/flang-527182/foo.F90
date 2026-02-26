module foo

  implicit none
  public

  type foo_type
     real(4), allocatable, dimension(:) :: x
   contains
     procedure :: allocate   => allocate_foo
     procedure :: deallocate   => deallocate_foo
     procedure, nopass :: create_device
     procedure, nopass :: delete_device
  end type foo_type

  contains
    subroutine create_device(this)
      type(foo_type), intent(inout) :: this
      integer i
      !$omp target enter data map(alloc:this%x) if(allocated(this%x))

      ! initialize with a kenrel to 0
      !$omp target teams distribute parallel do
      do i=1,size(this%x,1)
         this%x(i)=0.0
      enddo
      !$omp end target teams distribute parallel do
    end subroutine create_device

    subroutine delete_device(this)
      type(foo_type), intent(inout) :: this
      !$omp target exit data map(delete:this%x) if(allocated(this%x))
    end subroutine delete_device

    subroutine allocate_foo(this, n)
      class(foo_type), intent(inout) :: this
      integer,         intent(in)    :: n
      allocate(this%x(n))
    end subroutine allocate_foo
    
    subroutine deallocate_foo(this)
      class(foo_type), intent(inout) :: this
      deallocate(this%x)
    end subroutine deallocate_foo
end module foo
