module mymod
  implicit none
  private

  public :: vadd
 
  interface
     module subroutine vadd(a, b, c, N)
        implicit none
        real :: a(N), b(N), c(N)
        integer :: N
     end subroutine vadd
  end interface
  
end module mymod
