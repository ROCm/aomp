!  interface
subroutine dec_arrayval(i, beta) bind(C)
!$omp declare target
        use ISO_C_BINDING
        integer (C_INT), dimension(*), intent(out) :: beta
        integer (C_INT), value :: i
        beta(i+1) = beta(i+1) - 1
end subroutine
