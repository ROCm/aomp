program omp_taskwait_depend_minimal
double precision, allocatable, dimension(:)  :: a
!$omp taskwait depend(in:a)
end program
