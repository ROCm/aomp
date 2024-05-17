module radiation_spartacus_sw
  public
 
contains
 
  subroutine solver_spartacus_sw()
    implicit none
    integer :: n
    n = sizeof(1.0)
  end subroutine solver_spartacus_sw
 
end module radiation_spartacus_sw
program foo
   print *,'passed'
end
