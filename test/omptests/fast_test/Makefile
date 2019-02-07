include ../Makefile.defs

static:
	@export LIBRARY_PATH
	@export LD_LIBRARY_PATH
	@rm -rf results && mkdir results && cd results && \
	$(cmdx) -o $$PWD/../fast_test.bin $$PWD/../main.cpp -ldl
	@echo " ---> Shared library tester built successfully"
	
run:
	@export LD_LIBRARY_PATH
	./fast_test.bin list
	