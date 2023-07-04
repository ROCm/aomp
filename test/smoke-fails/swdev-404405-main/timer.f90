module timer
    interface
        subroutine tstart() bind(C, name="start")
            implicit none

        end subroutine

        subroutine tstop() bind(C, name="stop")
            implicit none

        end subroutine

        function telapsed() bind(C, name="elapsed")
            implicit none
            double precision :: telapsed
        end function

    end interface

end module 
