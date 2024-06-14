include ../../Makefile.defs

TEST_SRC    := dynamic_module_load.c
TEST_BIN    := dynamic_module_load
TEST_SO     := dynamic_module_load.so

# We do not get double free detected in tcache 2 with clang-linker-wrapper
# which is used with --no-opaque-offload-linker
#FIXARG       := --no-opaque-offload-linker
FIXARG       :=

CLANG        ?= clang
OMP_BIN      = $(AOMP)/bin/$(CLANG)
CC           = $(OMP_BIN) $(VERBOSE)

run : $(TEST_BIN) 
	./$(TEST_BIN) ./$(TEST_SO)

$(TEST_BIN) : $(TEST_SRC) $(TEST_SO)
	$(CC) $(TEST_SRC) -ldl -o $(TEST_BIN)
$(TEST_SO) : $(TEST_SRC)
	$(CC) -fopenmp $(TARGET) $(FIXARG) -DSHARED -fPIC -shared $(TEST_SRC) -o $(TEST_SO)

clean: 
	rm -f $(TEST_SO) $(TEST_BIN)
