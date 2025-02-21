module descr
  implicit none
  interface print_descr
    subroutine print_descr_real(name, d)  bind(c)
      implicit none
      real d(..)
      character*(*) name
    end subroutine
    subroutine print_descr_double(name, d) bind(c)
      implicit none
      double precision d(..)
      character*(*) name
    end subroutine
  end interface

end module
