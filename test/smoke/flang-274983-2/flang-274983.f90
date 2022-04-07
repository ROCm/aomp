program test
    use omp_lib
    use f90deviceio
      integer :: i
      real, pointer  :: res, x(:)
      integer(8) :: xaddr,xaddr1

      allocate(x(128))

      !$omp target enter data  map(to:x)

      do i=1, 128
         x(i)=2
      enddo

      write(0, *) "before device_ptr ", loc(x(1))
      !$omp target data use_device_ptr(x)
         xaddr = loc(x(1))
         write(0, *) "in device_ptr", loc(x(1))
      !$omp end target data
      write(0,*) "Addr " , xaddr

     !$omp target map(from: xaddr1)
       if (omp_get_thread_num() .eq. 0) then
         call f90printl("In tagret data=", loc(x(1)))
         xaddr1 = loc(x(1))
       endif
     !$omp end target
      write(0,*) "Addr1 " , xaddr1

     !$omp target exit data map(from:x)
      nullify(x)
      if (xaddr1 .ne. xaddr) then
        write(0,*) "FAILED in_device_ptr", xaddr, xaddr1
      endif
end program test
