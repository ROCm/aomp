#!/bin/bash
# rocprofv3 messages go to rocprofv3.log.
# Print that log only when this run fails, and keep the failing status.

aomphip=$1
log=rocprofv3.log

exit_with_log() {
    status=$1
    if [ "$status" -ne 0 ]; then
        cat "$log"
        exit "$status"
    fi
}

"$aomphip/bin/rocprofv3" --output-format csv --kernel-trace --stats -- ./launch_latency 2>"$log"
exit_with_log $?

python3 printLatency.py
exit_with_log $?
