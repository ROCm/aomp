module test_m
    implicit none
    public test_type
    type test_type
        character, pointer :: p1(:)
    end type
end module
program loop_test
    use test_m
    implicit none
    integer   :: i
    character, pointer :: p2(:), C(:)
    type(test_type), target   :: obj
    allocate(obj%p1(100))
    allocate(C(100))
    do i=1, 100
        obj%p1(i)='b'
        C(i)='a'
    end do
    do i=1, 10
        write(*, *) "p1=", obj%p1(i)
    end do
    ! Extra pointer for current workaround
    allocate(p2(100))
    p2=>obj%p1
    !$OMP TARGET ENTER DATA MAP(TO: obj, obj%p1)
    !!$OMP TARGET ENTER DATA MAP(TO: p2)
    !$OMP TARGET PARALLEL DO MAP(FROM:C)
    do i=1, 100
        C(i)=obj%p1(i)
   !     C(i)=p2(i)
    end do
    !$OMP END TARGET PARALLEL DO
    !$OMP TARGET EXIT DATA MAP(RELEASE:obj, obj%p1)
    !!$OMP TARGET EXIT DATA MAP(RELEASE: p2)
    if (c(1) .ne. obj%p1(1)) then
        write(*,*)"ERROR: wrong answer"
        stop 2
    endif
    write(*, *) "SUCCESS: C= ", C(1)
    return
end program loop_test
