program quad
real(kind=8) :: a, b, c

a = 0.0
!$omp target map(tofrom:a) map(from:b,c,d)
b = acos(a)
c = b * 2
d = cos(c)
!$omp end target

print *, a
print *, b
print *, c
print *, d

if (d + 1.0 < 1e-10) then
    print *, "Success"
else
    print *, "FAIL"
    stop(1)
endif
end
