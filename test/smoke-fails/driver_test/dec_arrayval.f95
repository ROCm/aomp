!  interface
subroutine dec_arrayval(i, beta) bind(C)
use ISO_C_BINDING
!$omp declare target
        integer (C_INT), dimension(*), intent(out) :: beta
        integer (C_INT), value :: i
        beta(i+1) = beta(i+1) - 1
end subroutine
