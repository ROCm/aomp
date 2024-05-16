[ -f ./flang_sim_cpu ] && rm ./flang_sim_cpu
[ -f ./flang_sim.o ] && rm ./flang_sim.o
[ -f ./helper_fns_cpu.o ] && rm ./helper_fns_cpu.o

# choose which compiler to test, for cpu all these work.
#_f_compiler="gfortran"
#_f_compiler="$HOME/rocm/aomp/bin/flang"
_f_compiler="$HOME/rocm/trunk/bin/flang-new"
#_cpp_compiler="g++"
#_cpp_compiler="$HOME/rocm/aomp/bin/clang++"
_cpp_compiler="$HOME/rocm/trunk/bin/clang++"

_omp_args="-O2 -fopenmp --offload-arch=gfx908"

echo $_f_compiler $_omp_args -c flang_sim.f95 -o flang_sim.o
$_f_compiler $_omp_args -c flang_sim.f95 -o flang_sim.o
echo
echo $_cpp_compiler -c  helper_fns_cpu.cpp -o helper_fns_cpu.o
$_cpp_compiler -c  helper_fns_cpu.cpp -o helper_fns_cpu.o
echo
echo $_f_compiler $_omp_args flang_sim.o helper_fns_cpu.o -o flang_sim_cpu
$_f_compiler $_omp_args flang_sim.o helper_fns_cpu.o -o flang_sim_cpu

echo
echo ./flang_sim_cpu
./flang_sim_cpu
