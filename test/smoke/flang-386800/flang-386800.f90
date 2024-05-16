subroutine cstar_assumed_shape_arg_2d (string)
  character(len=*) string (:,:)
  integer i
  i = 0
end subroutine cstar_assumed_shape_arg_2d

program flang_data_location_bug
  interface
     subroutine cstar_assumed_shape_arg_2d (string)
       character(len=*) string (:,:)
     end subroutine cstar_assumed_shape_arg_2d
  end interface

  character(len=3)  c3(4,5)
  real*8, dimension (0:19),     target :: array1
  real*8, pointer, dimension(:)   :: pa1

  do i = 1,5
     do j = 1,4
        write (unit=c3(j,i),fmt='(I1,'','',I1)')j,i
     end do
  end do

  do i = 0,19
     array1(i) = i+1
  end do
  pa1 => array1(::2)

  call cstar_assumed_shape_arg_2d (c3)

end program flang_data_location_bug
