module module_interface
  implicit none
  private

  public :: myfunc

  interface myfunc
     module procedure module_func_impl
  end interface myfunc
  
  interface
     module subroutine module_func_impl(i)
       integer :: i
     end subroutine module_func_impl
  end interface
  
end module module_interface

  

  
