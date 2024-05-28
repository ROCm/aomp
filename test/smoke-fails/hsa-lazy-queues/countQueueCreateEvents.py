#!/usr/bin/env python3

import json
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('num_occurences', type=int, help='How many hsa_queue_create events are expected')

filename = 'results.json'
hsa_queue_create_name = 'hsa_queue_create'

def searchAndCount(TheJSON, args) -> None:
  TraceEvents = TheJSON['traceEvents']

  ExpectedNumOccurs = args.num_occurences
  NumOccurs = 0
  for e in TraceEvents:
    if len(e) == 0:
      continue

    if e['name'] == hsa_queue_create_name:
      NumOccurs += 1

  if NumOccurs != ExpectedNumOccurs:
      sys.exit(1)

if __name__ == '__main__':
  args = parser.parse_args()
  with open(filename, "r") as f:
    J = json.load(f)
    searchAndCount(J, args)
