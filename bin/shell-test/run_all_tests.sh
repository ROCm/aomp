#!/bin/bash
# Test runner for all shell unit tests in the aomp project

# Get script directory
realpath=$(realpath "$0")
thisdir=$(dirname "$realpath")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

export SHUNIT_TMPDIR="/tmp/aomp_unit_testing"

# create the temporary directory if it does not exist
mkdir -p "$SHUNIT_TMPDIR" || {
    echo -e "${RED}[FAIL]${NC} Could not create temporary directory: $SHUNIT_TMPDIR"
    exit 1
}

# Test results tracking
total_test_files=0
total_tests=0
passed_tests=0
failed_tests=0
failed_test_files=()

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "PASS") echo -e "${GREEN}[PASS]${NC} $message" ;;
        "FAIL") echo -e "${RED}[FAIL]${NC} $message" ;;
        "INFO") echo -e "${YELLOW}[INFO]${NC} $message" ;;
    esac
}

# Function to run a single test file
run_test_file() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .sh)
    
    print_status "INFO" "Running $test_name..."
    
    # Create temporary log file
    local log_file="$SHUNIT_TMPDIR/aomp_${test_name}_$$.log"
    
    # Set up test environment variables
    export SHUNIT_COLOR='none'
    
    # Check if test file is executable
    if [ ! -x "$test_file" ]; then
        chmod +x "$test_file" 2>/dev/null || {
            print_status "FAIL" "$test_name (not executable and cannot make executable)"
            failed_tests=$((failed_tests + 1))
            failed_test_files+=("$test_name")
            return 1
        }
    fi
    
    # Run the test with proper error handling
    local exit_code=0
    local test_count=0
    local assertion_count=0
    local failure_count=0
    local error_count=0
    
    # Execute the test file and capture both stdout and stderr
    if timeout 300s bash -o pipefail "$test_file" > "$log_file" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi
    
    # Parse shunit2 output to get test statistics
    if [ -f "$log_file" ]; then
        # Check for shunit2-specific patterns
        if grep -q "^Ran [0-9]* test" "$log_file" 2>/dev/null; then
            # Parse shunit2 summary line: "Ran X tests."
            test_count=$(grep "^Ran [0-9]* test" "$log_file" | sed 's/^Ran \([0-9]*\) test.*/\1/')
            
            # Check for OK status
            if grep -q "^OK$" "$log_file" 2>/dev/null; then
                exit_code=0
            fi
            
            # Check for FAILURES
            if grep -q "^FAILED" "$log_file" 2>/dev/null; then
                exit_code=1
                # Parse failure line: "FAILED (failures=X)"
                failure_count=$(grep "^FAILED" "$log_file" | sed 's/.*failures=\([0-9]*\).*/\1/' 2>/dev/null || echo "1")
            fi
            
            # Check for errors in the failure line: "FAILED (failures=X, errors=Y)"
            if grep -q "errors=" "$log_file" 2>/dev/null; then
                error_count=$(grep "errors=" "$log_file" | sed 's/.*errors=\([0-9]*\).*/\1/' 2>/dev/null || echo "0")
            fi
        else
            # Count individual assertion calls if no summary found
            assertion_count=$(grep -c "^ASSERT" "$log_file" 2>/dev/null || echo "0")
            
            # If no shunit2 summary but assertions found, consider it a basic test
            if [ $assertion_count -gt 0 ]; then
                test_count=$assertion_count
            fi
        fi
        
        # Check for specific error patterns
        if grep -q "shunit2.*not found\|shunit2.*No such file" "$log_file" 2>/dev/null; then
            print_status "FAIL" "$test_name (shunit2 not found)"
            echo "  Please install shunit2 or check the path in the test file"
            echo "  Expected locations: \$HOME/local/shunit2/shunit2, /usr/share/shunit2/shunit2"
        elif grep -q "aomp_common_vars.*not found\|aomp_common_vars.*No such file" "$log_file" 2>/dev/null; then
            print_status "FAIL" "$test_name (aomp_common_vars not found)"
            echo "  Check that aomp_common_vars exists in the expected location: $thisdir/../aomp_common_vars"
        elif [ $exit_code -eq 124 ]; then
            print_status "FAIL" "$test_name (timeout after 300 seconds)"
        elif [ $exit_code -ne 0 ] || [ $failure_count -gt 0 ] || [ $error_count -gt 0 ]; then
            # Test failed
            local failure_msg="$test_name"
            if [ $failure_count -gt 0 ] && [ $error_count -gt 0 ]; then
                failure_msg="$failure_msg (failures: $failure_count, errors: $error_count)"
            elif [ $failure_count -gt 0 ]; then
                failure_msg="$failure_msg (failures: $failure_count)"
            elif [ $error_count -gt 0 ]; then
                failure_msg="$failure_msg (errors: $error_count)"
            else
                failure_msg="$failure_msg (exit code: $exit_code)"
            fi
            
            print_status "FAIL" "$failure_msg"
            
            # Show relevant error information
            echo "  Error details:"
            
            # Show individual test failures if present
            if grep -q "ASSERT:" "$log_file"; then
                echo "    Failed assertions:"
                grep "ASSERT:" "$log_file" | sed 's/^/      /' | head -5
            fi
            
            # Show shunit2 failure summary if available
            if grep -q "FAILED" "$log_file"; then
                grep -A 3 -B 1 "^FAILED" "$log_file" | sed 's/^/    /'
            fi
            
            # Test failed - add to failed list
            failed_test_files+=("$test_name")
            failed_tests=$((failed_tests + failure_count))
            passed_tests=$((passed_tests + test_count - failure_count))
            total_tests=$((total_tests + test_count))

            # Keep log file for failed tests for debugging
            echo "    Log file preserved at: $log_file"
            # return 1
        else
            # Test passed
            local success_msg="$test_name"
            if [ $test_count -gt 0 ]; then
                success_msg="$success_msg ($test_count tests)"
            elif [ $assertion_count -gt 0 ]; then
                success_msg="$success_msg ($assertion_count assertions)"
            else
                success_msg="$success_msg (completed successfully)"
            fi
            
            print_status "PASS" "$success_msg"
            passed_tests=$((passed_tests + test_count))
            total_tests=$((total_tests + test_count))

            # Clean up log file for successful tests unless verbose mode
            if [ "${verbose:-0}" -eq 0 ]; then
                rm -f "$log_file"
            else
                echo "    Log file: $log_file"
            fi
        fi
    else
        print_status "FAIL" "$test_name (no output generated)"
        failed_test_files+=("$test_name")
        failed_tests=$((failed_tests + 1))
    fi
    total_test_files=$((total_test_files + 1))
}

# Function to check prerequisites
check_prerequisites() {
    print_status "INFO" "Checking prerequisites..."
    
    # Check if shunit2 is available
    if [ ! -f "$HOME/local/shunit2/shunit2" ]; then
        print_status "FAIL" "shunit2 not found at $HOME/local/shunit2/shunit2"
        echo "Please install shunit2 first:"
        echo "  mkdir -p $HOME/local"
        echo "  cd $HOME/local"
        echo "  git clone https://github.com/kward/shunit2.git"
        exit 1
    fi
    
    # Check if aomp_common_vars exists
    if [ ! -f "$thisdir/../aomp_common_vars" ]; then
        print_status "FAIL" "aomp_common_vars not found at $thisdir/../aomp_common_vars"
        exit 1
    fi
    
    print_status "PASS" "Prerequisites check completed"
}

# Function to discover test files
discover_tests() {
    local test_files=()
    local quiet_mode=${1:-false}
    
    # Find all test_*.sh files in current directory
    for file in "$thisdir"/test_*.sh; do
        if [ -f "$file" ] && [ "$file" != "$0" ]; then
            # Add to test files array
            test_files+=("$file")
        fi
    done
    
    if [ ${#test_files[@]} -eq 0 ]; then
        if [ "$quiet_mode" != "true" ]; then
            print_status "FAIL" "No test files found matching pattern test_*.sh"
        fi
        exit 1
    fi
    
    if [ "$quiet_mode" != "true" ]; then
        print_status "INFO" "Found ${#test_files[@]} test files"
    fi
    
    printf '%s\n' "${test_files[@]}"
}

# Function to print final summary
print_summary() {
    echo
    echo "========================================="
    echo "TEST SUMMARY"
    echo "========================================="
    echo "Total test files: $total_test_files"
    echo "Total tests: $total_tests"
    echo "Passed: $passed_tests"
    echo "Failed: $failed_tests"
    
    if [ $failed_tests -gt 0 ]; then
        echo
        echo "Failed test files:"
        for failed_file in "${failed_test_files[@]}"; do
            echo "  - $failed_file"
        done
        echo
        print_status "FAIL" "Some tests failed"
        return 1
    else
        echo
        print_status "PASS" "All tests passed!"
        return 0
    fi
}

# Main execution
main() {
    echo "AOMP Shell Test Runner"
    echo "======================"
    echo
    
    # Parse command line arguments
    local verbose=0
    local pattern=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                verbose=1
                shift
                ;;
            -p|--pattern)
                pattern="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  -v, --verbose    Verbose output"
                echo "  -p, --pattern    Run tests matching pattern"
                echo "  -h, --help       Show this help"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    echo
    
    # Discover test files (with info messages)
    discover_tests
    echo
    
    # Get clean file list for processing
    local test_files
    mapfile -t test_files < <(discover_tests true)
    
    # Filter by pattern if provided
    if [ -n "$pattern" ]; then
        local filtered_files=()
        for file in "${test_files[@]}"; do
            if [[ "$(basename "$file")" == *"$pattern"* ]]; then
                filtered_files+=("$file")
            fi
        done
        test_files=("${filtered_files[@]}")
        
        if [ ${#test_files[@]} -eq 0 ]; then
            print_status "FAIL" "No test files match pattern: $pattern"
            exit 1
        fi
        
        print_status "INFO" "Filtered to ${#test_files[@]} test files matching '$pattern'"
    fi
    
    # Run each test file
    for test_file in "${test_files[@]}"; do
        print_status "INFO" "Running test file: $test_file"
        run_test_file "$test_file"
    done
    
    # Print summary and exit with appropriate code
    print_summary
}

# Run main function with all arguments
main "$@"
