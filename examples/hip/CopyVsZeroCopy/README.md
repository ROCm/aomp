CopyVsZeroCopy - Demonstrate performance difference on MI300A between Copy and Zero-Copy configurations.
=======================================================
This test is used to monitor performance difference between OpenMP's matching Copy and Zero-Copy configurations when
programmed in HIP.

To build in Copy configuration, use:
HSA_XNACK=0 make run

To build in Zero-Copy configuration, use
HSA_XNACK=1 make run

