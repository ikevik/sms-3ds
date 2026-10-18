#!/usr/bin/env python3
"""Names the classes in a log's [prof] lines (profile runs, platform/3ds/prof_3ds.cpp).

Usage: scripts/prof_names.py [sms3ds.elf] < log.txt
Each "vt XXXXXXXX" (an object's vtable pointer) gets the class whose vtable holds it, from the
ELF's _ZTV symbols. Other lines are left out.
"""
import bisect
import os
import re
import subprocess
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
elf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "sms3ds.elf")
nm = os.path.join(os.environ.get("DEVKITARM", "/opt/devkitpro/devkitARM"), "bin", "arm-none-eabi-nm")

tables = []  # (start, end, class)
out = subprocess.run([nm, "-C", "-S", elf], capture_output=True, text=True, check=True).stdout
for line in out.splitlines():
    parts = line.split(None, 3)
    if len(parts) == 4 and parts[3].startswith("vtable for "):
        start, size = int(parts[0], 16), int(parts[1], 16)
        tables.append((start, start + size, parts[3][len("vtable for "):]))
tables.sort()
starts = [t[0] for t in tables]


def name(addr):
    i = bisect.bisect_right(starts, addr) - 1
    if i >= 0 and tables[i][0] <= addr < tables[i][1]:
        return tables[i][2]
    return "?"


for line in sys.stdin:
    if not line.startswith("[prof]"):
        continue
    line = re.sub(r"vt ([0-9a-f]{8})", lambda m: "%-34s" % name(int(m.group(1), 16))[:34], line.rstrip("\n"))
    print(line)
