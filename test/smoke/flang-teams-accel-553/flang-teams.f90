program maino

  INTEGER, PARAMETER      :: x_min=1,x_max=10,y_min=1,y_max=10
  INTEGER, PARAMETER      :: number_of_states= 12
  INTEGER      :: g_rect
  INTEGER      :: g_circ
  INTEGER      :: g_point

  REAL(KIND=8) :: radius,x_cent,y_cent
  INTEGER      :: state

  INTEGER      :: j,k,jt,kt

!$omp target map(to:g_rect,g_circ,g_point,state_density,state_energy,    &
!$omp&           state_xvel,state_yvel,state_xmin,state_xmax,state_ymin, &
!$omp&           state_ymax,state_radius,state_geometry)
!$omp teams
!$omp distribute parallel do simd 
    DO j=x_min-2,x_max+2
 !    energy0(j,k)=state_energy(1)
    ENDDO
 !$omp distribute parallel do  private(radius,kt,jt) 
      DO j=x_min-2,x_max+2
     !  IF(state_geometry(state).EQ.g_rect ) THEN
        IF(1.EQ.g_rect ) THEN
        ENDIF
      ENDDO

!$omp end teams
!$omp end target

        end program maino
