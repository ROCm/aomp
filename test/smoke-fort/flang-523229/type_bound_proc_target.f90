! Reproducer for tickets SWDEV-523229 and SWDEV-540611.
! Compilation would fail while lowering to LLVM IR due to leftover host
! operations in the device MLIR module.

module mymodule
  implicit none

  type :: myclass
  contains
    procedure :: myfunc => myfunc
  end type myclass

contains
  subroutine myfunc(self)
    class(myclass) :: self
  end subroutine
end module

program main
  use mymodule, only : myclass
  implicit none

  class(myclass), allocatable :: x
  allocate(x)

  call x%myfunc()
  !$omp target
  !$omp end target

  deallocate(x)
end program
