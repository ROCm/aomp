.PHONY: build run clean

build: main

main: main.f03 hip_implementation.cpp
	hipfc $(CFLAGS) main.f03 hip_implementation.cpp -o main -lroctx64

run: main
	$(AOMPHIP)/bin/rocprofv3 --output-format csv --marker-trace --runtime-trace --stats -- ./main

clean:
	rm -rf main *.o *.mod rocprof_out `hostname`
