! LCOMPILER-2195: do concurrent reductions other than +, *, .and., .or.,
! .eqv., .neqv. all collapsed onto a single "other_reduction_<type>" symbol
! in flang lowering, so when more than one of {min, max, iand, ior, ieor}
! appeared in a translation unit the first one's combiner/init was reused
! for all of them. This test exercises each reduction kind and compares
! the GPU result against a sequential reference computed on the host.

program dc_reduce_test
  use iso_fortran_env, only: real64
  implicit none
  integer, parameter :: n = 1000000
  real(real64), allocatable :: a(:)
  real(real64) :: sum_seq, sum_par
  real(real64) :: minvalue, minvalue2
  real(real64) :: maxvalue, maxvalue2
  integer, allocatable :: ia(:)
  integer :: iand_seq, iand_par
  integer :: ior_seq,  ior_par
  integer :: ieor_seq, ieor_par
  integer :: i
  integer :: failures

  allocate(a(n))
  call random_number(a)
  a = a * 100.0_real64

  allocate(ia(n))
  do i = 1, n
    ia(i) = i
  end do

  ! Sequential reference values
  sum_seq = 0.0_real64
  do i = 1, n
    sum_seq = sum_seq + a(i)
  end do

  minvalue2 = HUGE(0.0_real64)
  do i = 1, n
    if (a(i) .gt. 0.5_real64) minvalue2 = MIN(minvalue2, a(i))
  end do

  maxvalue2 = 0.0_real64
  do i = 1, n
    if (a(i) .lt. 0.5_real64) maxvalue2 = MAX(maxvalue2, a(i))
  end do

  iand_seq = -1
  do i = 1, n
    iand_seq = iand(iand_seq, ia(i))
  end do

  ior_seq = 0
  do i = 1, n
    ior_seq = ior(ior_seq, ia(i))
  end do

  ieor_seq = 0
  do i = 1, n
    ieor_seq = ieor(ieor_seq, ia(i))
  end do

  !$omp target enter data map(to: a, ia)

  sum_par = 0.0_real64
  do concurrent (i = 1:n) reduce(+:sum_par)
    sum_par = sum_par + a(i)
  end do

  minvalue = HUGE(0.0_real64)
  do concurrent (i = 1:n) reduce(min:minvalue)
    if (a(i) .gt. 0.5_real64) minvalue = MIN(minvalue, a(i))
  end do

  maxvalue = 0.0_real64
  do concurrent (i = 1:n) reduce(max:maxvalue)
    if (a(i) .lt. 0.5_real64) maxvalue = MAX(maxvalue, a(i))
  end do

  iand_par = -1
  do concurrent (i = 1:n) reduce(iand:iand_par)
    iand_par = iand(iand_par, ia(i))
  end do

  ior_par = 0
  do concurrent (i = 1:n) reduce(ior:ior_par)
    ior_par = ior(ior_par, ia(i))
  end do

  ieor_par = 0
  do concurrent (i = 1:n) reduce(ieor:ieor_par)
    ieor_par = ieor(ieor_par, ia(i))
  end do

  !$omp target exit data map(from: a, ia)

  print '(a,1pg20.12)', 'Sequential sum   = ', sum_seq
  print '(a,1pg20.12)', 'DC reduce sum    = ', sum_par
  print '(a,1pg11.3)',  'Relative error   = ', abs(sum_par - sum_seq)/sum_seq
  print '(a,1pg20.12)', 'Min value serial = ', minvalue2
  print '(a,1pg20.12)', 'Min value gpu    = ', minvalue
  print '(a,1pg20.12)', 'Max value serial = ', maxvalue2
  print '(a,1pg20.12)', 'Max value gpu    = ', maxvalue
  print '(a,i12)',      'IAND serial      = ', iand_seq
  print '(a,i12)',      'IAND gpu         = ', iand_par
  print '(a,i12)',      'IOR  serial      = ', ior_seq
  print '(a,i12)',      'IOR  gpu         = ', ior_par
  print '(a,i12)',      'IEOR serial      = ', ieor_seq
  print '(a,i12)',      'IEOR gpu         = ', ieor_par

  failures = 0
  if (abs(sum_par - sum_seq)/sum_seq > 1.0e-10_real64) failures = failures + 1
  if (minvalue /= minvalue2)                            failures = failures + 1
  if (maxvalue /= maxvalue2)                            failures = failures + 1
  if (iand_par /= iand_seq)                             failures = failures + 1
  if (ior_par  /= ior_seq)                              failures = failures + 1
  if (ieor_par /= ieor_seq)                             failures = failures + 1

  if (failures /= 0) then
    print '(a,i0,a)', 'FAIL: ', failures, ' reduction kind(s) gave wrong result'
    stop 1
  end if

  print '(a)', 'PASS'

end program
