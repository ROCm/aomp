PROGRAM test
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTRDIFF_T
  IF (C_PTRDIFF_T >= 0) THEN
     PRINT "('Ok')"
  ELSE
     PRINT "('Not ok')"
  END IF
END program test
