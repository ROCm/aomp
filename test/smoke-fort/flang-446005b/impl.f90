submodule (module_interface) module_impl
  implicit none
 
contains

  module subroutine module_func_impl(i)
    integer :: i
    write(*,*) i 
  end subroutine module_func_impl
  
end submodule module_impl




  
