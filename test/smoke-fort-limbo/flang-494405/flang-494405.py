# Copyright © Advanced Micro Devices, Inc., or its affiliates.

# SPDX-License-Identifier:  MIT

# flang-494405.py
# LLM: ChatGPT-4o (OpenAI), 2024-06-08
#
# User query (included as documentation):
#
# In the directory where this program was executed, there is a directory with the current machine's name.
# In that directory, there is a file called: "*_kernel_stats.csv", for example: "1428133_kernel_stats.csv".  Read that file.
# This CSV file has the following format (for example):
#
# "Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
# "__omp_offloading_10302_2d8735b_glsc3_acc__l5",1,85531,85531.000000,100.00,85531,85531,0.00000000e+00
#
# Print out the information collected in the following format. What follows is an example:
#
# "Name"                                             "Calls"  "TotalDurationNs"  "AverageNs"  "Percentage"
# "__omp_offloading_10302_2d8735b_glsc3_acc__l5.kd"  1        447886             447886       100.0
#

import os
import socket
import glob
import csv

def main():
    # Get the current machine's hostname
    machine_name = socket.gethostname()
    # Build the path to the subdirectory named after the machine
    subdir = os.path.join(os.getcwd(), machine_name)

    if not os.path.isdir(subdir):
        print(f'No directory named {machine_name} in current working directory.')
        return

    # Search for *_kernel_stats.csv in the directory
    csv_files = glob.glob(os.path.join(subdir, '*_kernel_stats.csv'))
    if not csv_files:
        print(f'No *_kernel_stats.csv file found in {subdir}.')
        return

    csv_path = csv_files[0]

    # Read CSV file
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    if not rows:
        print('No data found in CSV file.')
        return

    # Prepare and print header
    header_fmt = '{:<52} {:<7} {:<16} {:<12} {:<10}'
    print(header_fmt.format(
        '"Name"', '"Calls"', '"TotalDurationNs"', '"AverageNs"', '"Percentage"'
    ))

    # Print each row in requested format, appending ".kd" to Name
    row_fmt = '{:<52} {:<7} {:<16} {:<12} {:<10}'
    for row in rows:
        name = f'{row["Name"]}.kd'
        calls = row["Calls"]
        total = int(float(row["TotalDurationNs"]))
        avg = int(float(row["AverageNs"]))
        perc = round(float(row["Percentage"]), 1)
        print(row_fmt.format(
            f'"{name}"', calls, total, avg, perc
        ))

if __name__ == '__main__':
    main()
