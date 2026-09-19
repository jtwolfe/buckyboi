import time
import sys
sys.path.insert(0, ".")
from sim import initial, step, projected_edges

s = initial()
mx, my = 400.0, 300.0
for _ in range(10_000):
    s = step(s, mx, my, False)
    projected_edges(s)

n = 200_000
t0 = time.perf_counter()
for _ in range(n):
    s = step(s, mx, my, False)
    projected_edges(s)
secs = time.perf_counter() - t0
print(f"python steps+project n={n} s={secs:.6f} per_s={n/secs:.0f} ns={secs*1e9/n:.1f}")
