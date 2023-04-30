program maino

  INTEGER, PARAMETER      :: x_min=1,x_max=10,y_min=1,y_max=10
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3) :: vertexx
  REAL(KIND=8), DIMENSION(y_min-2:y_max+3) :: vertexy
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2) :: cellx
  REAL(KIND=8), DIMENSION(y_min-2:y_max+2) :: celly
  REAL(KIND=8), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: density0,energy0
  REAL(KIND=8), DIMENSION(x_min-2:x_max+3,y_min-2:y_max+3) :: xvel0,yvel0
  INTEGER, PARAMETER      :: number_of_states= 12
  REAL(KIND=8), DIMENSION(number_of_states) :: state_density
  REAL(KIND=8), DIMENSION(number_of_states) :: state_energy
  REAL(KIND=8), DIMENSION(number_of_states) :: state_xvel
  REAL(KIND=8), DIMENSION(number_of_states) :: state_yvel
  REAL(KIND=8), DIMENSION(number_of_states) :: state_xmin
  REAL(KIND=8), DIMENSION(number_of_states) :: state_xmax
  REAL(KIND=8), DIMENSION(number_of_states) :: state_ymin
  REAL(KIND=8), DIMENSION(number_of_states) :: state_ymax
  REAL(KIND=8), DIMENSION(number_of_states) :: state_radius
  INTEGER     , DIMENSION(number_of_states) :: state_geometry
  INTEGER      :: g_rect
  INTEGER      :: g_circ
  INTEGER      :: g_point

  REAL(KIND=8) :: radius,x_cent,y_cent
  INTEGER      :: state

  INTEGER      :: j,k,jt,kt

  ! State 1 is always the background state

!$omp target map(to:g_rect,g_circ,g_point,state_density,state_energy,    &
!$omp&           state_xvel,state_yvel,state_xmin,state_xmax,state_ymin, &
!$omp&           state_ymax,state_radius,state_geometry)
!$omp teams
!$omp distribute parallel do simd 
    DO j=x_min-2,x_max+2
      energy0(j,k)=state_energy(1)
    ENDDO
 !$omp distribute parallel do  private(radius,kt,jt) 
      DO j=x_min-2,x_max+2
        IF(state_geometry(state).EQ.g_rect ) THEN
        ENDIF
      ENDDO

!$omp end teams
!$omp end target

        end program maino
