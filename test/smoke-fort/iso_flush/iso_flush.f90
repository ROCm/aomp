program iso_flush
    use, intrinsic :: iso_fortran_env, only : &
        & stdin=>input_unit, stdout=>output_unit, stderr=>error_unit

    print *, "Hello ISO"
    flush(stdout)
    flush(stderr)
end
