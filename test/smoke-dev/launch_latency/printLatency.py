#!/usr/bin/env python3
#
# LLM: AMD Nabu (GPT-4, 2024-06-12)
# Date: 2024-06-12
#
# Query:
# Write a Python program specified by the following.  If an error, return a negative error code. 
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
# Delete the directory with the current machine's name.

import os
import sys
import platform
import glob
import csv
import shutil
from collections import defaultdict

def main():
    try:
        machine_name = platform.node()
        if not machine_name:
            print("Error: Could not determine machine name.", file=sys.stderr)
            return -2

        machine_dir = os.path.join(os.getcwd(), machine_name)
        if not os.path.isdir(machine_dir):
            print(f"Error: Directory '{machine_dir}' does not exist.", file=sys.stderr)
            return -3

        csv_pattern = os.path.join(machine_dir, "*_kernel_trace.csv")
        csv_files = glob.glob(csv_pattern)
        if not csv_files:
            print(f"Error: No *_kernel_trace.csv found in {machine_dir}", file=sys.stderr)
            return -4

        csv_file = csv_files[0]

        kernel_records = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    kind = row["Kind"]
                    if kind != "KERNEL_DISPATCH":
                        continue
                    wsx = int(row["Workgroup_Size_X"])
                    wsy = int(row["Workgroup_Size_Y"])
                    wsz = int(row["Workgroup_Size_Z"])
                    gsx = int(row["Grid_Size_X"])
                    gsy = int(row["Grid_Size_Y"])
                    gsz = int(row["Grid_Size_Z"])
                    start = int(row["Start_Timestamp"])
                    end = int(row["End_Timestamp"])
                except Exception as e:
                    print(f"Error: Malformed row ({e})", file=sys.stderr)
                    return -5

                if wsx == 0 or wsy == 0 or wsz == 0:
                    print(f"Error: Workgroup size is zero in row", file=sys.stderr)
                    return -6

                teams = (gsx * gsy * gsz) // (wsx * wsy * wsz)
                exec_time_ns = end - start
                kernel_records.append((exec_time_ns, teams))

        if not kernel_records:
            print("Error: No valid kernel records found.", file=sys.stderr)
            return -7

        # Print first kernel time in seconds
        first_kernel_time_sec = kernel_records[0][0] / 1e9
        print(f"1st kernel Time {first_kernel_time_sec:.9f} seconds")

        # Aggregate by TEAMS and average
        teams_data = defaultdict(list)
        for exec_time_ns, teams in kernel_records:
            teams_data[teams].append(exec_time_ns)

        for teams in sorted(teams_data):
            times = teams_data[teams]
            avg_ns = sum(times) / len(times)
            print(f"avg kernel Time {avg_ns/1e9:.9f} seconds TEAMS= {teams}")

        # Remove the machine directory
        try:
            shutil.rmtree(machine_dir)
        except Exception as e:
            print(f"Error: Failed to delete directory {machine_dir}: {e}", file=sys.stderr)
            return -8

        return 0

    except Exception as e:
        print(f"Unhandled exception: {e}", file=sys.stderr)
        return -9

if __name__ == '__main__':
    sys.exit(main())
