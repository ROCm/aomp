module worker
 
    use, intrinsic :: iso_c_binding
    implicit none
#if 1
    integer, parameter :: TypeID = C_SHORT
#else
    integer, parameter :: TypeID = C_INT
#endif
 
    type SidreBuffer
    contains
        procedure :: get_type_id => buffer_get_type_id
    end type SidreBuffer
 
contains
 
   function buffer_get_type_id(obj) &
            result(SHT_rv)
        class(SidreBuffer) :: obj
        integer(TypeID) :: SHT_rv
        SHT_rv = 0
    end function buffer_get_type_id
 
end module worker
program main
          print *,'PASS'
          return
end
