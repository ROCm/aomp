#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The canonical rocKE CI result line consumed by extract-rocke.sh:
#
#   ROCKE_RESULT|<group>|<subtest>|<status>|<message>|<relevance>
#
# A newline in the message is encoded as US (0x1F) and a literal '|' as RS (0x1E),
# so a multi-line message survives as one line; extract-rocke.sh restores both.
#
# <relevance> says whether a red row can be a compiler-of-the-day regression at
# all; see rocke_relevance.py for the vocabulary. Empty when a lane has none.

from __future__ import annotations

import os
import sys

# In-message sentinels (see module header). rocke_extract.py decodes with these.
NL = "\x1f"
PIPE = "\x1e"


def sanitize(text: str) -> str:
    # rstrip real whitespace *before* encoding '|' (0x1E is whitespace to
    # str.rstrip, so encoding first would silently drop a trailing pipe).
    lines = [
        line.rstrip()
        for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return NL.join(lines).replace("|", PIPE)


def _record(line: str) -> None:
    """Copy a row to the worker's row log, when it asked for one.

    Lets run_rocke.sh tally its own run for the human summary without capturing
    stdout. Best-effort by design: a tally is a convenience and must never turn
    a good result into a failure.
    """
    path = os.environ.get("ROCKE_ROW_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def emit(
    group: str, subtest: str, status: int, message: str = "", relevance: str = ""
) -> None:
    line = (
        f"ROCKE_RESULT|{sanitize(group)}|{sanitize(subtest)}|{status}"
        f"|{sanitize(message)}|{sanitize(relevance)}"
    )
    print(line)
    sys.stdout.flush()
    _record(line)
