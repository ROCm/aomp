!  interface
!omp declare target
subroutine dec_arrayval(i, beta) bind(C)
        use ISO_C_BINDING
        integer (C_LONG), dimension(*), intent(out) :: beta
        integer (C_INT), value :: i
        beta(i) = beta(i) - 1
end subroutine
