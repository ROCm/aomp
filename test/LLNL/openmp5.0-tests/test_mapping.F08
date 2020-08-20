! Compile : xlf2008_r -qoffload -O3 -qsmp -qtgtarch=sm_70 test.F08
Program test
implicit none
type record
integer a
real(kind=kind(0.0d00)) :: r

real(kind=kind(0.0d00)),pointer :: data(:)
end type record
integer,parameter :: N=100
type(record) A

real(kind=kind(0.0d00)),allocatable :: alldata(:)
integer i
real(kind=kind(0.0d00)),target ::  adata(N)
real(kind(0.0d00)) :: sum


! Test a derived type with pointer
A%a=1
A%r=10.0d00
A%data=>adata
!$omp target map (tofrom: A)
!$omp parallel do
do i=1,N
A%data(i)=i**2+i
end do
!$omp end target



sum = 0.0d00
do i=1,N
   sum=sum+A%data(i)-i*i
end do


write(*,*)A%a,A%r

if (int(sum).EQ.(N*(N+1)/2)) then
   write(*,*)"Pointer test :: Success !!" 
else 
   write(*,*)"Pointer test :: FAIL",sum,"!=",N*(N+1)/2
endif


! Test with an allocatable array

allocate(alldata(N))
!$omp target map (tofrom: adata)
!$omp parallel do
do i=1,N
alldata(i)=i**2+i
end do
!$omp end target

sum = 0.0d00
do i=1,N
   sum=sum+alldata(i)-i*i
end do

if (int(sum).EQ.(N*(N+1)/2)) then
   write(*,*)"Allocatable test :: Success !!" 
else 
   write(*,*)"Allocatable test:: FAIL",sum,"!=",N*(N+1)/2
endif

deallocate(alldata)
end Program test
