PROGRAM reproducer
    IMPLICIT NONE

    REAL, ALLOCATABLE, TARGET, DIMENSION(:) :: arr

    INTEGER, PARAMETER :: ngrids = 2
    INTEGER, PARAMETER :: cellsperdim = 4
    INTEGER, PARAMETER :: cellspergrid = cellsperdim**3
    INTEGER :: iouter, ip3

    ALLOCATE(arr(ngrids * cellspergrid), source=-1.0)

    !$omp target teams distribute private(ip3) map(tofrom: arr)
    DO iouter = 1, ngrids
        ip3 = (iouter - 1) * cellspergrid + 1
        CALL kernel(arr(ip3))
    END DO
    !$omp end target teams distribute

    PRINT *, arr
    DEALLOCATE(arr)
CONTAINS
    SUBROUTINE kernel(gridarr)
        !$omp declare target

        ! Subroutine arguments
        REAL, INTENT(INOUT), DIMENSION(cellsperdim, cellsperdim, cellsperdim) :: gridarr

        ! Local variables
        INTEGER :: i, j, k

        !$omp parallel do collapse(2) private(i, j, k) shared(gridarr)
        DO i = 1, cellsperdim
            DO j = 1, cellsperdim
                DO k = 1, cellsperdim
                    gridarr(k, j, i) = REAL(i)
                END DO
            END DO
        END DO
        !$omp end parallel do
    END SUBROUTINE kernel
END PROGRAM reproducer
