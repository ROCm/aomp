#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import csv
import glob
import os
import shutil
import socket
import sys


FIELDS = ("Name", "Calls", "TotalDurationNs", "AverageNs", "Percentage")


def print_row(row):
    print(f'{"Name":<50} {"Calls":<8} {"TotalDurationNs":<18} {"AverageNs":<12} {"Percentage"}')
    print(
        f'{row["Name"]:<50} {row["Calls"]:<8} '
        f'{row["TotalDurationNs"]:<18} {row["AverageNs"]:<12} {row["Percentage"]}'
    )


def main():
    hostname_dir = os.path.join(os.getcwd(), socket.gethostname())
    csv_files = glob.glob(os.path.join(hostname_dir, "*_kernel_stats.csv"))
    if not csv_files:
        print(f"Error: no *_kernel_stats.csv found in {hostname_dir}", file=sys.stderr)
        return 1

    with open(csv_files[0], newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or any(field not in reader.fieldnames for field in FIELDS):
            print("Error: missing expected kernel stats columns", file=sys.stderr)
            return 1

        rows = [row for row in reader if row.get("Name")]
        if not rows:
            print("Error: no kernel rows found", file=sys.stderr)
            return 1

    slowest = max(rows, key=lambda row: float(row["TotalDurationNs"]))
    print_row(slowest)

    shutil.rmtree(hostname_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
