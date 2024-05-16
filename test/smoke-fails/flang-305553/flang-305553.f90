module fort2c
    USE ISO_C_BINDING
    interface
        subroutine saxpy_hip(N,A,B) BIND(C,name="saxpy_hip")
            USE ISO_C_BINDING
            INTEGER(C_INT),VALUE    :: N
            TYPE(C_PTR),VALUE       :: A,B
        end subroutine
    end interface
end module fort2c

subroutine copy(A,B,istart,iend,N)
    use ISO_C_BINDING
    use fort2c
    implicit none
    integer :: i,istart,iend,N,N2
    !real :: A(N), B(N) 
    real :: A(*), B(*)  

    !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO
    do i=istart,iend
        B(i) = A(i)
    end do
    !$OMP END TARGET TEAMS DISTRIBUTE PARALLEL DO
    
    
    N2 = iend - istart + 1
    
    !$OMP TARGET DATA USE_DEVICE_PTR(A,B)
    call saxpy_hip(N2,C_LOC(A(istart)),C_LOC(B(istart)))
    !$OMP END TARGET DATA

end subroutine

program main
    implicit none
    integer   :: i
    integer,parameter   :: istart = 1, iend=32, N=1024
    real, pointer :: A(:),B(:)
    allocate(A(N))
    allocate(B(N))
    do i=1, N
        A(i)=i
    end do
    B=0
    do i=1,iend+3
        write(*, *) "A=", A(i)
    end do
    !$OMP TARGET ENTER DATA MAP(TO:A,B)
    call copy(A,B,istart,iend,N)
    !$OMP TARGET UPDATE FROM(B)
    !$OMP TARGET EXIT DATA MAP(DELETE:A,B)
    do i=1,iend+3
        write(*, *) "B=", B(i)
    end do
    deallocate(A)
    deallocate(B)

end program main
