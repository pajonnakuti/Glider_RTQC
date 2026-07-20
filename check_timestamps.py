#!/usr/bin/env python3
"""Check timestamps of binary files in combined_binary to detect contamination."""
import os
import sys
import datetime

try:
    import dbdreader
except ImportError:
    print("ERROR: dbdreader not installed")
    sys.exit(1)

data_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/t/glider_data/1126/combined_binary"

# Point dbdreader to the pipeline's own cache directory
cache_dir = os.path.join(os.path.dirname(data_dir), "cache")
if os.path.isdir(cache_dir):
    os.environ["DBDREADER_CACHEDIR"] = cache_dir
    print(f"Using cache: {cache_dir}")

os.chdir(data_dir)

dbd_files = sorted([f for f in os.listdir('.') if f.endswith(('.dbd', '.dcd'))])
print(f"Checking {len(dbd_files)} flight files in {data_dir}")
print()

# Check first and last timestamps of each file
early_files = []
normal_files = []
deploy_start = datetime.datetime(2025, 2, 3)
deploy_end = datetime.datetime(2025, 4, 8)
margin = datetime.timedelta(days=30)

for f in dbd_files[:10]:  # Just check first 10
    try:
        d = dbdreader.DBD(f)
        t = d.get('m_present_time')
        d.close()
        if len(t[0]) > 0:
            first_ts = datetime.datetime.utcfromtimestamp(t[0][0])
            last_ts = datetime.datetime.utcfromtimestamp(t[0][-1])
            if first_ts < deploy_start - margin:
                early_files.append((f, first_ts, last_ts))
            else:
                normal_files.append((f, first_ts, last_ts))
            print(f"  OK {f}: {first_ts} -> {last_ts}")
        else:
            print(f"  EMPTY {f}: no m_present_time data")
    except Exception as e:
        print(f"  ERR {f}: {type(e).__name__}: {e}")

print(f"Files BEFORE deployment window ({deploy_start - margin}):")
for f, first, last in early_files[:20]:
    print(f"  {f}: {first} -> {last}")
if len(early_files) > 20:
    print(f"  ... and {len(early_files) - 20} more")
print(f"\nTotal: {len(early_files)} pre-deployment files, {len(normal_files)} within-deployment files")

if early_files:
    print(f"\nEarliest file: {early_files[0][0]} starts at {early_files[0][1]}")
    print(f"Latest pre-deployment file: {early_files[-1][0]} ends at {early_files[-1][2]}")
    print(f"\nRECOMMENDATION: Delete these {len(early_files)} files from combined_binary/")
    print(f"They are from a different deployment/test session and contaminate the L0.")
