#
#  kokkos_build.mk : Build kokkos from source
#
mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include $(mkfile_dir)../inc/find_gpu_and_install_dir.mk

KOKKOS_PREFIX      ?= /work/$(USER)/kokkos
KOKKOS_REPO        := https://github.com/kokkos/kokkos.git
KOKKOS_BRANCH      := develop
KOKKOS_TAG         := 3.7.01
KOKKOS_SHA         := 7c76889
supported_arch_vega :="900 906 908 90a"
supported_arch_navi :="1030 1100"
KOKKOS_BE          := omp
KOKKOS_GIT_DIR     := $(KOKKOS_PREFIX)/git
KOKKOS_BUILD_DIR   := $(KOKKOS_PREFIX)/kokkos-$(KOKKOS_TAG)_build_$(KOKKOS_BE).$(LLVM_GPU_ARCH)
KOKKOS_INSTALL_DIR := $(KOKKOS_PREFIX)/kokkos-$(KOKKOS_TAG)_$(KOKKOS_BE).$(LLVM_GPU_ARCH)
KOKKOS_LIB         := $(KOKKOS_INSTALL_DIR)/lib/libkokkoscore.a
_gfx_id := 90A
_arch_type := VEGA
KOKKOS_ARCH_ARG    := -D Kokkos_ARCH_$(_arch_type)$(_gfx_id)=ON 
KOKKOS_CMAKE_ARGS  := -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=17 -D CMAKE_CXX_EXTENSIONS=OFF -D CMAKE_INSTALL_PREFIX=$(KOKKOS_INSTALL_DIR) -D CMAKE_CXX_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang++ -D CMAKE_VERBOSE_MAKEFILE=ON $(KOKKOS_ARCH_ARG) -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ENABLE_OPENMPTARGET=ON -D Kokkos_ENABLE_COMPILER_WARNINGS=ON -D Kokkos_ENABLE_TESTS=ON
$(info +------------------------------------------------------------------------------+)
$(info | WARNING: Building Kokkos may take a very long time, make a new pot of coffee |)
$(info +------------------------------------------------------------------------------+)
$(info )

$(KOKKOS_LIB): $(KOKKOS_BUILD_DIR)/make_success
	cd $(KOKKOS_BUILD_DIR) ; make install >/dev/null

$(KOKKOS_GIT_DIR):
	mkdir -p $(KOKKOS_GIT_DIR)

$(KOKKOS_GIT_DIR)/kokkos/README.md: $(KOKKOS_GIT_DIR)
	cd $(KOKKOS_GIT_DIR) ; git clone $(KOKKOS_REPO) --branch $(KOKKOS_BRANCH)
	cd $(KOKKOS_GIT_DIR)/kokkos ; git checkout $(KOKKOS_TAG)

$(KOKKOS_GIT_DIR)/kokkos/is_patched: $(KOKKOS_GIT_DIR)/kokkos/README.md
	cd $(KOKKOS_GIT_DIR)/kokkos ; patch -p1 < $(mkfile_dir)kokkos.patch 
	touch $@

$(KOKKOS_BUILD_DIR)/cmake_success: $(KOKKOS_GIT_DIR)/kokkos/is_patched
	mkdir -p $(KOKKOS_BUILD_DIR) ; cd $(KOKKOS_BUILD_DIR) ; cmake $(KOKKOS_CMAKE_ARGS) $(KOKKOS_GIT_DIR)/kokkos
	touch $@

$(KOKKOS_BUILD_DIR)/make_success: $(KOKKOS_BUILD_DIR)/cmake_success
	cd $(KOKKOS_BUILD_DIR) ; make --output-sync=recurse -j8
	touch $@
