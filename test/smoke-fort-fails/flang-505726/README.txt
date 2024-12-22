/COD/LATEST/trunk-atd/bin/flang-new  -c  -fopenmp  -D__OFFLOAD_ARCH_gfx90a__ ttds.f90 -o ttds  

flang-new: /work1/omp-nightly/build/git/trunk20.0-atd/llvm-project/flang/lib/Lower/OpenMP/OpenMP.cpp:1320: void genBodyOfTargetOp(Fortran::lower::AbstractConverter&, Fortran::lower::SymMap&, Fortran::semantics::SemanticsContext&, Fortran::lower::pft::Evaluation&, mlir::omp::TargetOp&, const Fortran::common::openmp::EntryBlockArgs&, const mlir::Location&, const ConstructQueue&, llvm::SmallVectorImpl<tomp::DirectiveWithClauses<Fortran::lower::omp::Clause> >::const_iterator, Fortran::lower::omp::DataSharingProcessor&): Assertion `valOp != nullptr' failed.

