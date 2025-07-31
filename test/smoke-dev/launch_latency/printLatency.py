#!/usr/bin/env python3
# LLM: Nabu (GPT-4, June 2024) - 2024-06-09

"""
Query:

In the directory where this program was executed, there is a directory with the current machine's name.
In that directory, there is a file called: "*_kernel_trace.csv", for example: "759776_kernel_trace.csv".  Read that file.
This CSV file has the following format (for example):

"Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id","Kernel_Id","Kernel_Name",
 "Correlation_Id","Start_Timestamp","End_Timestamp","LDS_Block_Size","Scratch_Size","VGPR_Count",
 "Accum_VGPR_Count","SGPR_Count","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z",
 "Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
"KERNEL_DISPATCH","Agent 1",1,0,759776,1,1,"__omp_offloading_10302_2d86c1c_main_l19",1,
 1147738380281408,1147738380284052,0,0,4,4,16,256,1,1,256,1,1
...

Group all the different Grid_Size_X's together.  Maintain the order that the groups were encountered in the "*_kernel_trace.csv" file.  
For each different Grid_Size_X groups, sum their execution times.  
Each execution time is End_Timestamp - Start_Timestamp.  These units are nano-seconds.
The 1st group is the "1st kernel Time" referenced below.
The 2nd group is "TEAMS= 1"
The 3rd group is "TEAMS= 2"
and so on.

Print out the information collected in the following format.  Note the units are now in seconds.  What follows is an example:

1st kernel Time 0.000003680 seconds
avg kernel Time 0.000002235 seconds TEAMS= 1
avg kernel Time 0.000002284 seconds TEAMS= 2
avg kernel Time 0.000002293 seconds TEAMS= 4
...
avg kernel Time 0.000003024 seconds TEAMS= 512
avg kernel Time 0.000003670 seconds TEAMS= 1024
avg kernel Time 0.000005049 seconds TEAMS= 2048
"""

import os
import platform
import csv
import glob
from collections import OrderedDict

def find_machine_kernel_trace():
    machine = platform.node()
    # Look for files "*_kernel_trace.csv" inside the machine-named subdir
    kernel_trace_files = glob.glob(os.path.join(machine, "*_kernel_trace.csv"))
    if not kernel_trace_files:
        raise FileNotFoundError(f"No *_kernel_trace.csv found in directory '{machine}'")
    return kernel_trace_files[0]

def main():
    try:
        trace_csv = find_machine_kernel_trace()
    except Exception as e:
        print(e)
        return

    groups = OrderedDict()  # { Grid_Size_X: [duration_ns, count] }
    group_order = []
    with open(trace_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                grid_x = int(row['Grid_Size_X'])
                start = int(row['Start_Timestamp'])
                end = int(row['End_Timestamp'])
            except Exception:
                continue
            duration_ns = end - start
            if grid_x not in groups:
                groups[grid_x] = {"sum": 0, "count": 0}
                group_order.append(grid_x)
            groups[grid_x]["sum"] += duration_ns
            groups[grid_x]["count"] += 1

    if not group_order:
        print("No valid records found in file.")
        return

    for idx, grid_x in enumerate(group_order):
        total_ns = groups[grid_x]["sum"]
        cnt = groups[grid_x]["count"]
        avg_sec = total_ns / cnt / 1e9
        if idx == 0:
            print(f"1st kernel Time {avg_sec:.9f} seconds")
        else:
            print(f"avg kernel Time {avg_sec:.9f} seconds GRID_X = {grid_x}")

if __name__ == '__main__':
    main()
