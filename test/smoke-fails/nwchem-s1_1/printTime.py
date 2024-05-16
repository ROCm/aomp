# Python program to print the TotalDurationsNs in file: result.stats.csv.
# 
# Example:
#
# "Name","Calls","TotalDurationNs","AverageNs","Percentage"
# "__nv_tgt_sd_t_s1_1__TARGET_F1L23_1_.kd",1,4540333,4540333,100.0
#

f = open("results.stats.csv", "r")
for line in f:
    if line.startswith("\"Name\""):
        continue
    pList = line.split(",")
    TotalDurationNS = pList[2]
    print("Runtime: ", TotalDurationNS, "nanoseconds")
