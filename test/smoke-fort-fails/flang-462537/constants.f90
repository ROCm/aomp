module constants
implicit none
  real, parameter,private :: pi = 3.1415

contains
   subroutine print_pi()
      print*, "Pi = ", pi
   end subroutine print_pi

end module constants
