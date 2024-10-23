

program main
    implicit none
    integer :: nkgmax, nr, i, j, k
    real(kind=8), allocatable, dimension(:) :: cont_wave
    real(kind=8), allocatable, dimension(:,:) :: vmat2D
    real(kind=8) :: tmp
    integer :: time1, time2, dt, count_rate, count_max
    real(kind=8) :: secs_acc
    real(kind=8) :: vmat2d_min, vmat2d_max, vmat2d_mean, epsilon

    nkgmax = 10000
    nr = 10000
    epsilon = 1.0e-6

    allocate(cont_wave(1:nkgmax*nr))

    ! Initialize cont_wave
    cont_wave(:) = 0.d0
    do i = 1, nkgmax
        do j = 1, nr
            cont_wave((i-1)*nr + j) = dble(i - j) / dble(i + j)
        end do
    end do

    ! Allocate vmat2D
    allocate(vmat2D(1:nkgmax, 1:nkgmax))

    ! Start timing
    call system_clock(count_max=count_max, count_rate=count_rate)
    call system_clock(time1)

    print*, 'Before OpenMP Offload'

    ! OpenMP target offload
    !$omp target teams distribute parallel do num_teams(nkgmax) map(to: cont_wave) map(from: vmat2D)
    do i = 1, nkgmax
        !$omp simd
        do j = 1, nkgmax
            if (j > i) cycle
            tmp = 0.d0
            !!$omp simd
            do k = 1, nr
                tmp = tmp + cont_wave((k-1)*nkgmax + i) * cont_wave((k-1)*nkgmax + j)
            end do
            vmat2D(i, j) = tmp
            if (i /= j) vmat2D(j, i) = vmat2D(i, j)
        end do
    end do
    !$omp end target teams distribute parallel do

    ! End timing
    call system_clock(time2)
    dt = time2 - time1
    secs_acc = real(dt) / real(count_rate)
    print*, 'Time in secs with OpenMP Offload: ', secs_acc

    print*, 'Min value in vmat2D: ', minval(vmat2D(1:nkgmax, 1:nkgmax))
    print*, 'Max value in vmat2D: ', maxval(vmat2D(1:nkgmax, 1:nkgmax))
    print*, 'Mean value in vmat2D: ', sum(vmat2D(1:nkgmax, 1:nkgmax)) / dble(nkgmax * nkgmax)

    ! Time in secs with OpenMP Offload:  48.430999755859375
    ! Min value in vmat2D:  -3847.6374622142002
    ! Max value in vmat2D:  9967.42851222314
    ! Mean value in vmat2D:  1109.6848768110396

    vmat2d_min  = minval(vmat2D(1:nkgmax, 1:nkgmax))
    vmat2d_max  = maxval(vmat2D(1:nkgmax, 1:nkgmax))
    vmat2d_mean = sum(vmat2D(1:nkgmax, 1:nkgmax)) / dble(nkgmax * nkgmax)
    print *, abs((vmat2d_min  - -3847.6374622142002) / vmat2d_min )
    print *, abs((vmat2d_max  - 9967.42851222314   ) / vmat2d_max )
    print *, abs((vmat2d_mean - 1109.6848768110396 ) / vmat2d_mean)

    if (abs((vmat2d_min  - -3847.6374622142002) / vmat2d_min ) > epsilon) stop -1
    if (abs((vmat2d_max  - 9967.42851222314   ) / vmat2d_max ) > epsilon) stop -2
    if (abs((vmat2d_mean - 1109.6848768110396 ) / vmat2d_mean) > epsilon) stop -3

end program main

