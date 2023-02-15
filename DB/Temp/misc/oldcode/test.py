import os
import sys

n = int(sys.argv[1])
a = 2
tables = []
for _ in range(n):
    tables.append(sys.argv[a])
    a += 1
print(tables)
