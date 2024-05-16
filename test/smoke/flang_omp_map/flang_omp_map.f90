MODULE definitions_module

  IMPLICIT NONE
  TYPE field_type
    REAL(KIND=8),    DIMENSION(:,:), ALLOCATABLE :: density0,density1
  END TYPE field_type

  TYPE tile_type
    TYPE(field_type):: field
  END TYPE tile_type
   
  TYPE chunk_type
    TYPE(tile_type), DIMENSION(:), ALLOCATABLE :: tiles
  END TYPE chunk_type
  TYPE(chunk_type)       :: chunk
END MODULE definitions_module
SUBROUTINE start
  use definitions_module
  ALLOCATE(chunk%tiles (1))
  ALLOCATE(chunk%tiles(1)%field%density0  (-1:962,-1:962))
  ALLOCATE(chunk%tiles(1)%field%density1  (-1:962,-1:962))

 print '(Z)', loc(chunk%tiles(1))
 print '(Z)', loc(chunk%tiles(1)%field)
 print '(Z)', loc(chunk%tiles(1)%field%density0)
 print '(Z)', loc(chunk%tiles(1)%field%density1)

 call flush(6)
!$omp target data map(                   &
!$omp   chunk%tiles(1)%field%density0,   &
!$omp   chunk%tiles(1)%field%density1)

  DO tile=1,2
    chunk%tiles(1)%field%density0(tile,-1) =0
    chunk%tiles(1)%field%density1(tile,-1) =0
  ENDDO
!$omp end target data

END SUBROUTINE start
program foobar 
   call start
end program
