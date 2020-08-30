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
!$omp target data map(                   &
!$omp   chunk%tiles(1)%field%density0,   &
!$omp   chunk%tiles(1)%field%density1)

  DO tile=1,tiles_per_chunk
    CALL initialise_chunk(tile)
  ENDDO
!$omp end target data

END SUBROUTINE start
program foobar

end program
