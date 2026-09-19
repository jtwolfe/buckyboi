import time, sys
sys.path.insert(0, ".")
from sim import initial, step
s = initial()
for _ in range(10_000):
    s = step(s, 400.0, 300.0, False)
n = 1_000_000
t0 = time.perf_counter()
for _ in range(n):
    s = step(s, 400.0, 300.0, False)
secs = time.perf_counter() - t0
print(f"python step-only n={n} s={secs:.6f} per_s={n/secs:.0f} ns={secs*1e9/n:.1f}")
