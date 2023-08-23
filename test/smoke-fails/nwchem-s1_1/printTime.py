# Python program to print the TotalDurationsNs in file: result.stats.csv.
# 
# Example:
#
# "Name","Calls","TotalDurationNs","AverageNs","Percentage"
# "__nv_tgt_sd_t_s1_1__TARGET_F1L23_1_.kd",1,4540333,4540333,100.0
#

f = open("results.stats.csv", "r")
index = 0
for line in f:
    index = index + 1
    pList = line.split(",")
    if index == 2:
        print(pList[2])
