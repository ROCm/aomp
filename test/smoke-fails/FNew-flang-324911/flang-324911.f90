program target_offload
    implicit none
    integer :: upper
    upper = 10
    call distribute_parallel_do(upper)
    print *, "PASS"
    return
contains
subroutine distribute_parallel_do(upper)
    use OMP_LIB
    integer upper
    integer :: a(1:upper), b(1:upper), c(1:upper), i

    a = 13
    b = 61
    c = 0

    !$OMP target parallel do map(to:a,b), map(from:c)
    do i = 1, upper ! [in-omp-region]
        c(i) = a(i) * b(i)
    end do

    do i = 1, upper
      if (c(i) .ne. 793) then
        write(*,*) "FAIL: Wrong answer c =", c, "Expected answer : 793"
        stop 2
      endif
    end do

end subroutine distribute_parallel_do
end program
