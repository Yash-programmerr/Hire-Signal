import subprocess
import time
import psutil
import sys

RANK_COMMAND = ["python", "rank.py", "--candidates", "./candidates.jsonl", "--out", "./submission.csv"]
TIME_LIMIT_SEC = 5 * 60
MEM_LIMIT_GB = 16

print(f"Running: {' '.join(RANK_COMMAND)}")
start = time.time()
proc = subprocess.Popen(RANK_COMMAND)
ps_proc = psutil.Process(proc.pid)

peak_mem_gb = 0
while proc.poll() is None:
    try:
        mem = ps_proc.memory_info().rss
        for child in ps_proc.children(recursive=True):
            mem += child.memory_info().rss
        peak_mem_gb = max(peak_mem_gb, mem / 1e9)
    except psutil.NoSuchProcess:
        pass
    time.sleep(0.5)

elapsed = time.time() - start
returncode = proc.returncode

print(f"\n{'='*50}")
print(f"Elapsed time : {elapsed:.1f}s  (limit: {TIME_LIMIT_SEC}s)")
print(f"Peak memory  : {peak_mem_gb:.2f} GB  (limit: {MEM_LIMIT_GB} GB)")
print(f"Exit code    : {returncode}")

errors = []
if elapsed > TIME_LIMIT_SEC:
    errors.append(f"OVER TIME LIMIT by {elapsed - TIME_LIMIT_SEC:.1f}s")
if peak_mem_gb > MEM_LIMIT_GB:
    errors.append(f"OVER MEMORY LIMIT by {peak_mem_gb - MEM_LIMIT_GB:.2f}GB")
if returncode != 0:
    errors.append("rank.py exited with non-zero code")

if errors:
    print("\nFAILED compute compliance:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("\nCompute compliance PASSED")