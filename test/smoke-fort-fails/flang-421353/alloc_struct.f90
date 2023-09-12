module myindexer
type, public :: indexer
        character(len=:), allocatable :: escape, ignore_between
        contains

    end type indexer
    end module

program struct_test
        use myindexer
        character, parameter :: escape = "\\", ignore_between = '"'
        type(indexer) index_
        index_ = indexer(escape, ignore_between)  
        end program
