#!/usr/bin/env python3

import json
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('num_occurences', type=int, help='How many hsa_queue_create events are expected')

filename = 'results.json_results.json'
hsa_queue_create_name = 'hsa_queue_create'

def searchAndCount(TheJSON, args) -> None:
  Data = TheJSON['rocprofiler-sdk-tool']
  for D in Data:
    if 'summary' in D:
      Summary = D['summary']

  for E in Summary:
    if E['domain'] != 'HSA_API':
      continue

    Stats = E['stats']
    Ops = Stats['operations']
    for Op in Ops:
      if Op['key'] == hsa_queue_create_name:
        NumOccur = Op['value']['count']
        # Return error if numbers don't match
        sys.exit(NumOccur - args.num_occurences)

  # When not found
  sys.exit(1)

if __name__ == '__main__':
  args = parser.parse_args()
  print("Reading JSON file:", filename)
  with open(filename, "r") as f:
    J = json.load(f)
    searchAndCount(J, args)
