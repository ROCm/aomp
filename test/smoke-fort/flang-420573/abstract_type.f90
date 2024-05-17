! compile with
!
! flang-new -flang-experimental-hlfir -flang-experimental-polymorphism -o abstract_type.o -c abstract_type.f90

module my_mod
    implicit none

    type, public, abstract :: class_type
        private
        procedure(procA), pointer :: procA => null()
        contains
            procedure,public :: procB => procB_impl
    end type class_type

    abstract interface
        subroutine procA(me, a, b, c)
            import :: class_type
            implicit none
            class(class_type), intent(inout) :: me
            real(kind=8) :: a, b, c
        end subroutine procA
    end interface

    contains
        subroutine procB_impl(me)
            implicit none
            class(class_type), intent(inout) :: me
        end subroutine procB_impl
end module my_mod
