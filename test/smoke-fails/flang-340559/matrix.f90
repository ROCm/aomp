module matrix
  implicit none
  private

  type, public :: matrix_t
    integer*4 :: row, col
    real*8, pointer :: val(:,:)
  end type

  type, public :: hmatrix_t
    integer*4 :: cnt
    type(matrix_t), pointer :: mat(:), vec(:), res(:)
  end type
end module
