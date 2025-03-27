module foo

  implicit none
  public

  type foo_type
     real(4), allocatable, dimension(:) :: x
   contains
     procedure :: allocate   => allocate_foo
     procedure :: deallocate   => deallocate_foo
     procedure :: create_device
     procedure :: delete_device
  end type foo_type

  contains
    subroutine create_device(this)
      class(foo_type), intent(inout) :: this
      !$OMP TARGET ENTER DATA MAP(TO:this%x) IF(allocated(this%x))
    end subroutine create_device
    
    subroutine delete_device(this)
      class(foo_type), intent(inout) :: this
      !$OMP TARGET EXIT DATA MAP(DELETE:this%x) IF(allocated(this%x)) 
    end subroutine delete_device

    subroutine allocate_foo(this, N)
      class(foo_type), intent(inout) :: this
      integer,         intent(in)    :: N
      allocate(this%x(N))
    end subroutine allocate_foo
    
    subroutine deallocate_foo(this)
      class(foo_type), intent(inout) :: this
      deallocate(this%x)
    end subroutine deallocate_foo
end module foo

program main
  use foo, only: foo_type
  implicit none
  type(foo_type) :: bar

  call bar%allocate(10)
  !$omp target enter data map(to:bar)
  call bar%create_device()
  call bar%delete_device()  
  !$omp target exit data map(from:bar)
  call bar%deallocate()
  
end program main
