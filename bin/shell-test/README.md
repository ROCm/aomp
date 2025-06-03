# AOMP Shell Unit Tests

This directory contains unit tests for the AOMP shell scripts.

## Prerequisites

1. **shunit2**: The testing framework used for unit tests
   ```bash
   mkdir -p $HOME/local
   cd $HOME/local
   git clone https://github.com/kward/shunit2.git
   ```

2. **AOMP environment**: Ensure AOMP environment variables are properly set

## Running Tests

### Run All Tests
```bash
# Run all tests in the directory
./run_all_tests.sh

# Run with verbose output
./run_all_tests.sh --verbose
```

### Run Specific Tests
```bash
# Run tests matching a pattern
./run_all_tests.sh --pattern "common"

# Run a specific test file
./test_aomp_common_vars.sh
```

### Test Files

- `test_aomp_common_vars.sh` - Tests for aomp_common_vars script
- `test_build_aomp.sh` - Tests for build_aomp.sh script  
- `test_build_project.sh` - Tests for build_project.sh script
- `test_build_openmp.sh` - Tests for build_openmp.sh script

### Test Structure

Each test file follows this pattern:
1. Defines test functions following `test_*` naming convention
2. Uses shunit2 assertions (assertEquals, assertNotNull, etc.)
3. Includes shunit2 at the end to run the tests

## Example:
```bash
#!/bin/bash
# Unit tests for my_script.sh

realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")

test_my_function() {
    local result=$(my_function "test_input")
    assertEquals "Expected output" "test_input_processed" "$result"
}

. $HOME/local/shunit2/shunit2
```

## Continuous Integration

The test runner returns:
- Exit code 0 if all tests pass
- Exit code 1 if any tests fail

This makes it suitable for CI/CD pipelines.
