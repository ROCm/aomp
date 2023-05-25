#!/usr/bin/env python3

import json
import sys

filename = 'results.json'
hsa_queue_create_name = 'hsa_queue_create'

def searchAndCount(TheJSON) -> None:
  TraceEvents = TheJSON['traceEvents']

  NumOccurs = 0
  for e in TraceEvents:
    if len(e) == 0:
      continue

    if e['name'] == hsa_queue_create_name:
      NumOccurs += 1

  if NumOccurs < 2:
      sys.exit(1)

if __name__ == '__main__':
  with open(filename, "r") as f:
    J = json.load(f)
    searchAndCount(J)
