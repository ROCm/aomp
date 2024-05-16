module matrix
  implicit none
  private
  public construct_hmatrix, destruct_hmatrix, fill_hmatrix, copy_hmatrix, is_hmatrix_equal, &
         allocate_matrix, deallocate_matrix, fill_matrix, &
         matrix_product_on_cpu, matrix_product_on_gpu, is_matrix_equal

  type, public :: matrix_t
    integer*4 :: row, col
    real*8, pointer :: val(:,:)
  end type

  type, public :: hmatrix_t
    integer*4 :: cnt
    type(matrix_t), pointer :: mat(:), vec(:), res(:)
  end type

contains

  type(hmatrix_t) function construct_hmatrix(cnt) result(hmat)
    implicit none
    integer*4 :: cnt, i, row, col

    hmat%cnt = cnt
    row = 10; col = 2048
    allocate(hmat%mat(cnt), hmat%vec(cnt), hmat%res(cnt))
    do i = 1,cnt
      call allocate_matrix(hmat%mat(i), row, col)
      call allocate_matrix(hmat%vec(i), col, 1)
      call allocate_matrix(hmat%res(i), row, 1)
    enddo
  end function

  subroutine destruct_hmatrix(hmat)
    implicit none
    type(hmatrix_t) :: hmat
    integer*4 :: i

    do i = 1,hmat%cnt
      call deallocate_matrix(hmat%mat(i))
      call deallocate_matrix(hmat%vec(i))
      call deallocate_matrix(hmat%res(i))
    enddo
    deallocate(hmat%mat, hmat%vec, hmat%res)
    hmat%cnt = 0
  end subroutine

  subroutine fill_hmatrix(hmat)
    implicit none
    type(hmatrix_t) :: hmat
    integer*4 :: i

    do i = 1,hmat%cnt
      call fill_matrix(hmat%mat(i))
      call fill_matrix(hmat%vec(i))
    enddo
  end subroutine

  type(hmatrix_t) function copy_hmatrix(src) result(dst)
    implicit none
    type(hmatrix_t) :: src
    integer*4 :: i, row, col

    dst = construct_hmatrix(src%cnt)
    do i = 1,src%cnt
      row = src%mat(i)%row; col = src%mat(i)%col
      dst%mat(i)%val(1:col,1:row) = src%mat(i)%val(1:col,1:row)
      dst%vec(i)%val(1,1:col) = src%vec(i)%val(1,1:col)
    enddo
  end function

  subroutine allocate_matrix(mat, row, col)
    implicit none
    type(matrix_t) :: mat
    integer*4 :: row, col, r, c

    mat%row = row; mat%col = col
    allocate(mat%val(col,row))
    do r = 1,row
      do c = 1,col
        mat%val(c,r) = 0.0
      enddo
    enddo
  end subroutine

  subroutine deallocate_matrix(mat)
    implicit none
    type(matrix_t) :: mat

    deallocate(mat%val)
    mat%row = 0; mat%col = 0
  end subroutine

  subroutine fill_matrix(mat)
    implicit none
    type(matrix_t) :: mat

    call random_seed()
    call random_number(mat%val)
  end subroutine

  subroutine matrix_product_on_cpu(hmat)
    implicit none
    type(hmatrix_t) :: hmat
    type(matrix_t), pointer :: mat, vec, res
    real*8 :: curr
    integer*4 :: idx, row, col, r, c

    write(*,*) 'subroutine matrix_product_on_cpu start!'
    do idx = 1,hmat%cnt
      mat => hmat%mat(idx); vec => hmat%vec(idx); res => hmat%res(idx)
      row = mat%row; col = mat%col
      do r = 1,row
          curr = 0.0d0
          do c = 1,col
            curr = curr + mat%val(c,r) * vec%val(1,c)
          enddo
          res%val(1,r) = res%val(1,r) + curr
      enddo
    enddo
  end subroutine

  subroutine matrix_product_on_gpu(hmat)
    implicit none
    type(hmatrix_t) :: hmat
    type(matrix_t), pointer :: mat, vec, res
    real*8 :: curr
    integer*4 :: idx, row, col, r, c
    real*8, pointer :: mat_d(:,:), vec_d(:,:), res_d(:,:)

    write(*,*) 'subroutine matrix_product_on_gpu start!'
    do idx = 1,hmat%cnt
      mat => hmat%mat(idx); vec => hmat%vec(idx); res => hmat%res(idx)
      mat_d => mat%val; vec_d => vec%val; res_d => res%val
      row = mat%row; col = mat%col
      !$omp target map(to: mat%val, vec%val) map(tofrom: res%val)
      !!$omp target map(to: mat_d, vec_d) map(tofrom: res_d)
      do r = 1,row
          curr = 0.0d0
          !$omp parallel do reduction(+:curr)
          do c = 1,col
            curr = curr + mat%val(c,r) * vec%val(1,c)
            !curr = curr + mat_d(c,r) * vec_d(1,c)
          enddo
          res%val(1,r) = res%val(1,r) + curr
          !res_d(1,r) = res_d(1,r) + curr
      enddo
      !$omp end target
    enddo
  end subroutine

  logical function is_matrix_equal(m1, m2) result(res)
    implicit none
    type(matrix_t) :: m1, m2
    integer*4 :: r, c, r1, r2, c1, c2

    res = .true.
    r1 = m1%row; c1 = m1%col
    r2 = m2%row; c2 = m2%col
    if ((r1 .ne. r2) .or. (c1 .ne. c2)) then
      res = .false.
      return
    endif

    do c = 1,c1
      do r = 1,r1
        if (abs(m1%val(c,r) - m2%val(c,r)) >= 1.0d-6) then
          write(*,*) 'Error at (', c, ',', r, ')'
          write(*,*) 'm1: ', m1%val(c,r)
          write(*,*) 'm2: ', m2%val(c,r)
          res = .false.
          return
        endif
      enddo
    enddo
  end function

  logical function is_hmatrix_equal(hm1, hm2) result(res)
    implicit none
    type(hmatrix_t) :: hm1, hm2
    integer*4 :: idx

    write(*,*) 'subroutine is_hmatrix_equal start!'
    res = .true.
    if (hm1%cnt .ne. hm2%cnt) then
      write(*,*) 'Error: hmatrix1%cnt != hmatrix2%cnt'
      res = .false.
      return
    endif

    do idx = 1,hm1%cnt
      res = is_matrix_equal(hm1%mat(idx), hm2%mat(idx))
      if (.not. res) then
        write(*,*) 'Error: matrix ', idx, ' not equal'
        return
      endif
      res = is_matrix_equal(hm1%vec(idx), hm2%vec(idx))
      if (.not. res) then
        write(*,*) 'Error: vector ', idx, ' not equal'
        return
      endif
      res = is_matrix_equal(hm1%res(idx), hm2%res(idx))
      if (.not. res) then
        write(*,*) 'Error: res ', idx, ' not equal'
        return
      endif
    enddo
  end function
end module
