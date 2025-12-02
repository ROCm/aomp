#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.

# SPDX-License-Identifier:  MIT

"""
LLM: Claude 3.5 Sonnet
Date: December 2024

Query: Write a Python program specified by the following. If an error, return a negative error code.

In a comment in the program, specify the LLM/version and the current date. Include this query into the program as documentation.

In the directory where this program was executed, there is a directory with the current machine's name.
In that directory, there is a file called: "*_kernel_stats.csv", for example: "1428133_kernel_stats.csv". Read that file.
This CSV file has the following format (for example):

"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"__omp_offloading_10302_2d8735b_glsc3_acc__l5",1,85531,85531.000000,100.00,85531,85531,0.00000000e+00

Print out the information collected in the following format. What follows is an example:

"Name"                                             "Calls"  "TotalDurationNs"  "AverageNs"  "Percentage"
"__omp_offloading_10302_2d8735b_glsc3_acc__l5.kd"  1        447886             447886       100.0

Delete the directory with the current machine's name.
"""

import os
import csv
import glob
import shutil
import socket
import sys

def main():
    try:
        # Get current machine's hostname
        machine_name = socket.gethostname()

        # Check if directory exists
        if not os.path.isdir(machine_name):
            print(f"Error: Directory '{machine_name}' not found.")
            return -1

        # Find the *_kernel_stats.csv file inside the directory
        pattern = os.path.join(machine_name, "*_kernel_stats.csv")
        csv_files = glob.glob(pattern)

        if not csv_files:
            print(f"Error: No '*_kernel_stats.csv' file found in '{machine_name}'.")
            return -2

        if len(csv_files) > 1:
            print(f"Error: Multiple '*_kernel_stats.csv' files found in '{machine_name}'.")
            return -3

        csv_file = csv_files[0]

        # Read CSV and print formatted output
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)

            # Print header
            print(f'{"Name":<50} {"Calls":<8} {"TotalDurationNs":<18} {"AverageNs":<12} {"Percentage"}')

            for row in reader:
                name = row["Name"] + ".kd"
                calls = row["Calls"]
                total_duration = row["TotalDurationNs"]
                average_ns = int(float(row["AverageNs"]))
                percentage = float(row["Percentage"])

                print(f'{name:<50} {calls:<8} {total_duration:<18} {average_ns:<12} {percentage}')

        # Delete the directory with the machine name
        shutil.rmtree(machine_name)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return -4

if __name__ == "__main__":
    sys.exit(main())
