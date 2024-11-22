program use_device_test
   use iso_c_binding
   interface
      type(c_ptr) function get_ptr() BIND(C)
         USE, intrinsic :: iso_c_binding
         implicit none
      end function get_ptr

      integer(c_int) function check_result(host, dev) BIND(C)
         USE, intrinsic :: iso_c_binding
         implicit none
         type(c_ptr), intent(in) :: host, dev
      end function check_result
   end interface

   type(c_ptr) :: device_ptr, x

   x = get_ptr()
   device_ptr = x

   !$omp target data map(tofrom: x) use_device_ptr(x)
   device_ptr = x
   !$omp end target data

   if (check_result(x, device_ptr) .ne. 0) then
      stop 1
   end if
end program use_device_test
