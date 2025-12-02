#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.

# SPDX-License-Identifier:  MIT

# flang-494405.py
# LLM: GPT-4
# Date: 2025-08-20
"""
Write a Python program specified by the following.  If an error, return a negative error code.

In a comment in the program, specify the LLM/version and the current date. Include this query into the program as documentation.

In the directory where this program was executed, there is a directory with the current machine's name.
In that directory, there is a file called: "*_kernel_stats.csv", for example: "1428133_kernel_stats.csv".  Read that file.
This CSV file has the following format (for example):

"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"__omp_offloading_10302_2d86ff6_tgt_sd_t_s1_1__l23.kd",1,888032,888032,100.0

Print out the information collected in the following format. What follows is an example:

"Name"                                             "Calls"  "TotalDurationNs"  "AverageNs"  "Percentage"
"__omp_offloading_10302_2d8735b_glsc3_acc__l5.kd"  1        447886             447886       100.0

Delete the directory with the current machine's name.
"""

import os
import sys
import socket
import glob
import csv
import shutil

def print_row(name, calls, total_ns, average_ns, percentage):
    # Format to match requested output: columns aligned similar to example
    print(f'{name:<50} {calls:<8} {total_ns:<18} {average_ns:<12} {percentage}')

def main():
    try:
        hostname = socket.gethostname()
        hostname_dir = os.path.join(os.getcwd(), hostname)

        if not os.path.isdir(hostname_dir):
            print(f"Error: directory '{hostname}' not found in current directory.", file=sys.stderr)
            return -1

        pattern = os.path.join(hostname_dir, "*_kernel_stats.csv")
        files = glob.glob(pattern)
        if not files:
            print(f"Error: no file matching '*_kernel_stats.csv' in '{hostname}'.", file=sys.stderr)
            return -2

        csv_file = files[0]

        # Read CSV and print rows
        with open(csv_file, newline='') as f:
            # The CSV may have quoted header; use csv.DictReader to be robust
            reader = csv.DictReader(f)
            # Print header line as requested
            print(f'{"Name":<50} {"Calls":<8} {"TotalDurationNs":<18} {"AverageNs":<12} {"Percentage"}')

            # Validate required columns present
            required = ("Name", "Calls", "TotalDurationNs", "AverageNs", "Percentage")
            for col in required:
                if col not in reader.fieldnames:
                    print(f"Error: expected column '{col}' not found in CSV.", file=sys.stderr)
                    return -3

            for row in reader:
                raw_name = row.get("Name", "").strip()
                if not raw_name:
                    # skip empty names
                    continue
                # Ensure .kd suffix as in example
                name = raw_name if raw_name.endswith(".kd") else raw_name + ".kd"
                calls = row.get("Calls", "").strip()
                total_ns = row.get("TotalDurationNs", "").strip()
                # AverageNs may be float-like; format as integer when appropriate
                avg_raw = row.get("AverageNs", "").strip()
                try:
                    avg_val = int(float(avg_raw)) if avg_raw != "" else ""
                except ValueError:
                    avg_val = avg_raw
                perc_raw = row.get("Percentage", "").strip()
                try:
                    perc_val = float(perc_raw) if perc_raw != "" else ""
                except ValueError:
                    perc_val = perc_raw

                print_row(name, calls, total_ns, avg_val, perc_val)

        # Delete the hostname directory (recursively)
        try:
            shutil.rmtree(hostname_dir)
        except PermissionError:
            print(f"Error: permission denied when deleting directory '{hostname_dir}'.", file=sys.stderr)
            return -4
        except OSError as e:
            print(f"Error: failed to delete directory '{hostname_dir}': {e}", file=sys.stderr)
            return -5

        return 0

    except FileNotFoundError as e:
        print(f"Error: file not found: {e}", file=sys.stderr)
        return -6
    except PermissionError as e:
        print(f"Error: permission error: {e}", file=sys.stderr)
        return -7
    except Exception as e:
        print(f"Error: unexpected error: {e}", file=sys.stderr)
        return -8

if __name__ == "__main__":
    sys.exit(main())
