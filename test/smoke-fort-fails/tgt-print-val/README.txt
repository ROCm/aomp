3/10/25 now getting output with emissary I/O

Wrong answer with complex(kind=8):

    real(kind=4)    :: fval = 14
    real(kind=8)    :: dval = 18
    complex(kind=4) :: cfval = (24, 25)
    complex(kind=8) :: cdval = (28, 29)
    ...
    print *, fval, dval, cfval, cdval
    ...
    print *, dfval

< Seen
> Expected

3c3
<  14. 18. (24.,25.) (28.,25.)
---
>  14. 18. (24.,25.) (28.,29.)
7c7
<  (28.,1.1428102855502297E+243)
---
>  (28.,29.)

Previously:
Failure mode:
ld.lld: error: undefined symbol: _FortranAioBeginExternalListOutput
ld.lld: error: undefined symbol: _FortranAioOutputAscii
ld.lld: error: undefined symbol: _FortranAioOutputInteger32
ld.lld: error: undefined symbol: _FortranAioEndIoStatement
ld.lld: error: undefined symbol: _FortranAioOutputInteger8
ld.lld: error: undefined symbol: _FortranAioOutputInteger16
ld.lld: error: undefined symbol: _FortranAioOutputInteger64
ld.lld: error: undefined symbol: _FortranAioOutputReal32
ld.lld: error: undefined symbol: _FortranAioOutputReal64
ld.lld: error: undefined symbol: _FortranAioOutputComplex32
ld.lld: error: undefined symbol: _FortranAioOutputComplex64
