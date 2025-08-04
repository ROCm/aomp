#!/usr/bin/env python3
#
# printLatency.py
#
# Query:
# Write a Python program called printLatency.py.  If an error, return a negative error code.
#
# In a comment in the program, specify the LLM/version and the current date.
#
# Include this query into the program as documentation.  In a comment in the program, specify the LLM/version and the current date
#
# In the directory where this program was executed, there is a directory with the current machine's name.
# In that directory, there is a file called: "*_kernel_trace.csv", for example: "759776_kernel_trace.csv".  Read that file.
# This CSV file has the following format (for example):
#
# "Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
# "KERNEL_DISPATCH","Agent 1",1,0,759776,1,1,"__omp_offloading_10302_2d86c1c_main_l19",1,1147738380281408,1147738380284052,0,0,4,4,16,256,1,1,256,1,1
# ...
#
# The calculation for TEAMS is (Grid_Size_X * Grid_Size_Y*Grid_Size_Z) / (Workgroup_Size_X*Workgroup_Size_Y*Workgroup_Size_Z).
#
# For each TEAMS, sum their execution times.  Each execution time is End_Timestamp - Start_Timestamp.  These units are nano-seconds.
#
# Print out the information collected in the following format.  Note the units are now in seconds.  What follows is an example:
#
# 1st kernel Time 0.000003680 seconds
# avg kernel Time 0.000002235 seconds TEAMS= 1
# avg kernel Time 0.000002284 seconds TEAMS= 2
# avg kernel Time 0.000002293 seconds TEAMS= 4
# ...
# avg kernel Time 0.000003024 seconds TEAMS= 512
# avg kernel Time 0.000003670 seconds TEAMS= 1024
# avg kernel Time 0.000005049 seconds TEAMS= 2048
#
# LLM: AMD Nabu (GPT-4, 2024-06-12)
# Date: 2024-06-12

import os
import platform
import sys
import glob
import csv
from collections import defaultdict

def main():
    try:
        # Get machine name
        machine_name = platform.node()
        if not machine_name:
            print("ERROR: Could not determine machine name.", file=sys.stderr)
            return -2

        machine_dir = os.path.join(os.getcwd(), machine_name)
        if not os.path.isdir(machine_dir):
            print(f"ERROR: Directory '{machine_dir}' does not exist.", file=sys.stderr)
            return -3

        # Find *_kernel_trace.csv in the machine directory
        pattern = os.path.join(machine_dir, "*_kernel_trace.csv")
        files = glob.glob(pattern)
        if not files:
            print(f"ERROR: No *_kernel_trace.csv file found in '{machine_dir}'.", file=sys.stderr)
            return -4

        filename = files[0]

        # Prepare to read CSV and process records
        kernel_records = []
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    wsx = int(row["Workgroup_Size_X"])
                    wsy = int(row["Workgroup_Size_Y"])
                    wsz = int(row["Workgroup_Size_Z"])
                    gsx = int(row["Grid_Size_X"])
                    gsy = int(row["Grid_Size_Y"])
                    gsz = int(row["Grid_Size_Z"])
                    start = int(row["Start_Timestamp"])
                    end = int(row["End_Timestamp"])
                    kind = row["Kind"]
                    if kind != "KERNEL_DISPATCH":
                        continue
                except Exception as e:
                    print(f"ERROR: Malformed row: {e}", file=sys.stderr)
                    return -5

                teams = (gsx * gsy * gsz) // max(1, (wsx * wsy * wsz))
                exec_time_ns = end - start
                kernel_records.append( (exec_time_ns, teams) )

        if not kernel_records:
            print("ERROR: No kernel records found in file.", file=sys.stderr)
            return -6

        # Print 1st kernel execution time
        first_kernel_time_sec = kernel_records[0][0] / 1e9
        print(f"1st kernel Time {first_kernel_time_sec:.9f} seconds")

        # Aggregate execution times by TEAMS
        teams_stats = defaultdict(list)
        for exec_time_ns, teams in kernel_records:
            teams_stats[teams].append(exec_time_ns)

        for teams in sorted(teams_stats):
            times = teams_stats[teams]
            avg_sec = (sum(times) / len(times)) / 1e9
            print(f"avg kernel Time {avg_sec:.9f} seconds TEAMS= {teams}")

        return 0

    except FileNotFoundError:
        print("ERROR: File not found.", file=sys.stderr)
        return -7
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}", file=sys.stderr)
        return -8

if __name__ == '__main__':
    sys.exit(main())
