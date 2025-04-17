#!/usr/bin/env python3

import argparse
import typing
import subprocess
import shlex
import os

parser = argparse.ArgumentParser(prog='GTest TestCase Extractor')
parser.add_argument('executable', help='The name of the executable to query')


def process(gtest_output):
  split_out = gtest_output.split('\n')
  currTestSuite=''
  currTestCase=''
  tests = []
  for s in split_out:
    if s.find('.') != -1:
      currTestSuite=s[:s.find('.')]
      currTestCase = '' # Reset
    else:
      currTestCase=s.strip()
    if len(currTestSuite) > 0 and len(currTestCase) == 0:
      continue

    tests.append(currTestSuite + '.' + currTestCase)
  return tests

def myMain(args) -> None:
  sh_command = "{} --gtest_list_tests".format(args.executable)
  sh_args = shlex.split(sh_command)
  os.environ['OMP_NUM_THREADS']='2'
  os.environ['OMP_PROC_BIND']='spread'
  cp = subprocess.run(sh_args, shell=False, check=True, capture_output=True, text=True) # TODO: wire-up out/err streams
  output = cp.stdout

  tests_to_run = process(output)
  for t in tests_to_run:
    print(t)

if __name__ == "__main__":
  myMain(parser.parse_args())
