
// This is just a dummy.
int main(void) { return 0; }

// Note: The error message we are looking for should be:
//       /path/to/llvm-link: No such file or directory: 'MissingFile.bc'

/// CHECK: llvm-link
/// CHECK-SAME: No such file or directory:
/// CHECK-SAME: 'MissingFile.bc'
/// CHECK-NOT: {{.+}}
