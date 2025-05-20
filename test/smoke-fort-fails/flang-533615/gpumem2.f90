module mod
    implicit none
    type :: mattype
        real(4),pointer :: array(:,:,:)
        integer(4) :: scalar
    end type
    type :: data
        type(mattype) :: memb
    end type
contains
    subroutine us_gpumem(dat)
        implicit none
        type(data), pointer :: dat
        !$omp target enter data map(to:dat,dat%memb)
    end subroutine us_gpumem
end module mod
