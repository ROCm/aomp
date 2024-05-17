module trigger
 
  implicit none
    
  integer, parameter :: realkind = kind(0.0d0)
 
contains

#ifdef WITH_ARG
  subroutine trigger_func(an_arg)
#else
  subroutine trigger_func()
#endif
 
#ifdef WITH_ARG
    integer, dimension(*), intent(in)         :: an_arg
#endif

#ifdef WITH_ARR
    real(realkind), allocatable               :: an_arr(:)
    real(realkind), allocatable, dimension(:) :: another_arr
#endif

#ifdef WITH_CONST
    real(realkind), parameter                 :: a_const = 0.0
#endif
 
  end subroutine trigger_func
 
end module trigger

program main
        print  *,"compilation error previously, so passes"
end program main
