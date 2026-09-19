import time, sys
sys.path.insert(0, ".")
from sim import BG, WIN_H, WIN_W, initial, mix_wire, projected_edges, step

def plot(buf, x, y, color):
    if 0 <= x < WIN_W and 0 <= y < WIN_H:
        buf[y * WIN_W + x] = color

def draw_line(buf, x0, y0, x1, y1, color):
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        plot(buf, x, y, color)
        plot(buf, x + 1, y, color)
        plot(buf, x, y + 1, color)

buf = [BG] * (WIN_W * WIN_H)
state = initial()
sink = 0
for _ in range(10):
    state = step(state, 400.0, 300.0, False)
    for i in range(len(buf)):
        buf[i] = BG
    for ax, ay, bx, by, z in projected_edges(state):
        draw_line(buf, ax, ay, bx, by, mix_wire(z))
    sink ^= buf[100]

n = 120
t0 = time.perf_counter()
for _ in range(n):
    state = step(state, 400.0, 300.0, False)
    for i in range(len(buf)):
        buf[i] = BG
    for ax, ay, bx, by, z in projected_edges(state):
        draw_line(buf, ax, ay, bx, by, mix_wire(z))
    sink ^= buf[int(state.cx) % len(buf)]
secs = time.perf_counter() - t0
print(f"python softbuffer-frames n={n} s={secs:.4f} fps={n/secs:.1f} ms/frame={secs*1000/n:.3f} sink={sink}")
