# Python program to print the launch latency in file: result.csv.
#
# Example:
# 
# lstringe@r7:~/git/aomp18.0/aomp/test/smoke/launch_latency$ head results.csv
# "Index","KernelName","gpu-id","queue-id","queue-index","pid","tid","grd","wgr","lds","scr","arch_vgpr","accum_vgpr","sgpr","wave_size","sig","obj","DispatchNs","BeginNs","EndNs","CompleteNs","DurationNs"
# 0,"__omp_offloading_fd00_5871764_main_l19.kd",2,1,0,3495164,3495164,256,256,512,0,4,4,16,64,"0x153e64afca00","0x153d64770ec0",5323868593531200,5323868593552306,5323868593554226,5323868593566416,1920
# 1,"__omp_offloading_fd00_5871764_main_l28.kd",2,2,0,3495164,3495164,257,257,512,0,4,4,16,64,"0x153e64afca00","0x153d64770f00",5323868597609543,5323868597625664,5323868597648223,5323868597650689,22559
# 2,"__omp_offloading_fd00_5871764_main_l28.kd",2,3,0,3495164,3495164,257,257,512,0,4,4,16,64,"0x153e64afca00","0x153d64770f00",5323868601572039,5323868601587825,5323868601610545,5323868601613071,22720
# 3,"__omp_offloading_fd00_5871764_main_l28.kd",2,0,0,3495164,3495164,257,257,512,0,4,4,16,64,"0x153e64afca00","0x153d64770f00",5323868601623008,5323868601645584,5323868601668144,5323868601669341,22560
#

dict = {}
f = open("results.csv", "r")
for line in f:
    line = line.rstrip()  # strip off the carriage return and line feed at the end of line
    # print (line)
    pList = line.split(",")
    grd = pList[7]
    if grd == "\"grd\"": 
        continue
    durationNS = int(pList[-1])
    if grd in dict:
        count, sum = dict[grd] 
        count = count + 1
        sum = sum + durationNS
        dict[grd] = (count, sum)
    else:
        dict[grd] = (1, durationNS)

count,sum = dict["256"]
latencyaverage = (float(sum) / count) * 1e-9
print ("1st kernel Time", "{:11.9f} seconds".format(latencyaverage))
dict.pop("256")

j = 1
for key in dict:
    count, sum = dict[key]
    latencyaverage = (float(sum) / count) * 1e-9
    # "avg kernel Time %12.8f TEAMS=%d\n"
    print ("avg kernel Time", "{:11.9f} seconds".format(latencyaverage), "TEAMS=", j)
    j = j * 2

