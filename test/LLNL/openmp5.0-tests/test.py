#!/usr/bin/env python3
from collections import defaultdict
import subprocess
import sys

if len(sys.argv) != 3:
  print ("Error: Argument list is incorrect. Test.py expects AOMP location and AOMP_GPU.")
  exit(1)

AOMP = sys.argv[1]
AOMP_GPU = sys.argv[2]

if AOMP == "":
  print ("Error: Please set AOMP env variable and rerun.")
  exit(1)
elif AOMP_GPU == "":
  print ("Error: Please set AOMP_GPU env variable and rerun.")
  exit(1)

def get_tests(file_name):
    d=defaultdict(list)
    o=defaultdict(lambda: '')
    with open(file_name,"r") as infile:
        for line in infile.readlines():
            #print(line)
            if '/' in line:
                files,opts = line.split('/')
                sline=files.split()
                o[sline[0][:-4]]=opts
            else:
                sline=line.split()
            for s in sline:
                d[sline[0][:-4]].append(s)
    return d,o

def run(tests):
    pass_count=0
    for t in tests:
        passs=True
        print("Running ",t,"...",end=" ")
        cmd="./"+t
        cmd_s=cmd.split()
        p4=subprocess.Popen(cmd_s,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        stdout,stderr = p4.communicate()
        stdout = stdout.decode('utf-8')
        for lines in stdout.splitlines():
            #print(lines)
            if "error" in lines:
                print("Failed")
                passs=False
                break
            if "FAIL" in lines:
                print("Failed")
                passs=False
                break
        if p4.returncode != 0:
            passs=False
        if passs: 
            print ("Passed")
            pass_count+=1
    return pass_count

def compile(CC, LIBS, tests, opts):

    runnables=[]
    for key,value in tests.items():
        with open(key+".ERR","w") as efile:
            passs=True
            for v in value:
                fail=False
                extraopts=opts[key]
                cmd=CC+" -c " + v + " " + extraopts
                cmd_s=cmd.split()
                p2=subprocess.Popen(cmd_s,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                stdout,stderr = p2.communicate()
                stdout = stdout.decode('utf-8')
                for lines in stdout.splitlines():
                    efile.write(lines)
                    if "error" in lines:
                        fail=True
                        print("Compilation of ",v," failed")
                        passs = passs and not fail
                        break
                    passs = passs and not fail
                if p2.returncode != 0:
                    passs=False
 
            if passs:
                print("Compiling ",key)
                cmd = CC+" -o "+key
                for v in value:
                    cmd = cmd+" "+v[:-4]+".o"
                cmd = cmd+LIBS
                #print("Final compile command is ",cmd)
                cmd_s=cmd.split()
                p3=subprocess.Popen(cmd_s,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                stdout, stderr = p3.communicate()
                stdout = stdout.decode('utf-8')
                for lines in stdout.splitlines():
                    print(lines)
                    if "error" in lines:
                        print("Linking of ",v," failed\n")
                        break
                if p3.returncode == 0:
                    runnables.append(key)
    return runnables
def main():
    tests,opts=get_tests("test_list")
# Change Compile line in CC and LIBS
    CC="{}/bin/clang++  -O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march={}".format(AOMP, AOMP_GPU)
    LIBS = ""
# End Compile line 
    runnables=compile(CC, LIBS, tests, opts)
    print("\nRunnable tests are:")
    for r in runnables:
        print(r)
    print()
    pass_count=run(runnables)
    print("There are ",len(tests.keys())," tests")
    print(len(runnables)," tests compiled successfully")
    print(pass_count," tests ran successfully")
if __name__=="__main__":
	main()
