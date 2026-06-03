"""Print the number of finished games in a PGN file (count of [Result tags).

Used by tune_cpuct.bat to decide whether a matchup needs to be re-run. Prints
0 if the file does not exist, so the caller doesn't need to pre-check.
"""

import os
import sys

if len(sys.argv) != 2:
    print(0)
    sys.exit(0)

path = sys.argv[1]
if not os.path.exists(path):
    print(0)
    sys.exit(0)

count = 0
with open(path, "r", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if line.startswith("[Result "):
            count += 1
print(count)
