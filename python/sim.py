"""Shared icosahedron simulation. Constants and formulas match SPEC.md."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

# --- constants (keep in lockstep with rust/src/sim.rs and bend/main.bend) ---

WIN_W = 800
WIN_H = 600
MARGIN = 16
BG = 0x12141A
WIRE_NEAR = 0xF2F5FC
WIRE_FAR = 0x7A8194

PHI = (1.0 + math.sqrt(5.0)) / 2.0
_NORM = math.sqrt(1.0 + PHI * PHI)

ICO_SCALE = 72.0
FOCAL = 4.0
ROT_X = 0.014
ROT_Y = 0.021
BOUND_RADIUS = 104.0
LINE_HALF = 1.1

HIT_RADIUS = 100.0
REPEL_RADIUS = 180.0
REPEL_STRENGTH = 4.2
LOCK_LERP = 0.065
LOCK_EPS = 2.0
VEL_DAMP = 0.88
HOVER_PUSH = 6.5
HOVER_FRAMES = 12

LO_X = MARGIN + BOUND_RADIUS  # 120
HI_X = WIN_W - MARGIN - BOUND_RADIUS  # 680
LO_Y = MARGIN + BOUND_RADIUS  # 120
HI_Y = WIN_H - MARGIN - BOUND_RADIUS  # 480

CORNERS: Tuple[Tuple[float, float], ...] = (
    (LO_X, LO_Y),  # 0 TL
    (HI_X, LO_Y),  # 1 TR
    (HI_X, HI_Y),  # 2 BR
    (LO_X, HI_Y),  # 3 BL
)

# Unnormalized vertices, then unit-length.
_RAW = (
    (0.0, -1.0, -PHI),
    (0.0, -1.0, PHI),
    (0.0, 1.0, -PHI),
    (0.0, 1.0, PHI),
    (-1.0, -PHI, 0.0),
    (-1.0, PHI, 0.0),
    (1.0, -PHI, 0.0),
    (1.0, PHI, 0.0),
    (-PHI, 0.0, -1.0),
    (PHI, 0.0, -1.0),
    (-PHI, 0.0, 1.0),
    (PHI, 0.0, 1.0),
)
VERTS: Tuple[Tuple[float, float, float], ...] = tuple(
    (x / _NORM, y / _NORM, z / _NORM) for x, y, z in _RAW
)

EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 2),
    (0, 4),
    (0, 6),
    (0, 8),
    (0, 9),
    (1, 3),
    (1, 4),
    (1, 6),
    (1, 10),
    (1, 11),
    (2, 5),
    (2, 7),
    (2, 8),
    (2, 9),
    (3, 5),
    (3, 7),
    (3, 10),
    (3, 11),
    (4, 6),
    (4, 8),
    (4, 10),
    (5, 7),
    (5, 8),
    (5, 10),
    (6, 9),
    (6, 11),
    (7, 9),
    (7, 11),
    (8, 10),
    (9, 11),
)

IDLE_LOCKING = 0
LOCKED = 1
FLOATING = 2
SEEKING_LOCK = 3


@dataclass
class State:
    cx: float
    cy: float
    vx: float
    vy: float
    ax: float
    ay: float
    mode: int
    lock: int
    grabbed: bool
    ox: float
    oy: float
    hover: int
    mx: float
    my: float
    button: bool


def initial() -> State:
    return State(
        cx=400.0,
        cy=300.0,
        vx=0.0,
        vy=0.0,
        ax=0.0,
        ay=0.0,
        mode=IDLE_LOCKING,
        lock=0,
        grabbed=False,
        ox=0.0,
        oy=0.0,
        hover=0,
        mx=-1000.0,
        my=-1000.0,
        button=False,
    )


def corner(lock: int) -> Tuple[float, float]:
    return CORNERS[lock & 3]


def other_corner(lock: int, mx: float, my: float) -> int:
    """Farthest corner from the cursor, excluding `lock`. Ties: lowest index."""
    best = -1
    best_d2 = -1.0
    for i in range(4):
        if i == lock:
            continue
        tx, ty = CORNERS[i]
        d2 = (tx - mx) * (tx - mx) + (ty - my) * (ty - my)
        if d2 > best_d2:
            best_d2 = d2
            best = i
    return best


def _clamp(s: State) -> None:
    if s.cx < LO_X:
        s.cx = LO_X
        s.vx = 0.0
    if s.cx > HI_X:
        s.cx = HI_X
        s.vx = 0.0
    if s.cy < LO_Y:
        s.cy = LO_Y
        s.vy = 0.0
    if s.cy > HI_Y:
        s.cy = HI_Y
        s.vy = 0.0


def _repel(s: State) -> None:
    dx = s.mx - s.cx
    dy = s.my - s.cy
    d = math.hypot(dx, dy)
    if d <= 0.0 or d >= REPEL_RADIUS:
        return
    t = 1.0 - d / REPEL_RADIUS
    force = REPEL_STRENGTH * t * t
    s.vx -= force * dx / d
    s.vy -= force * dy / d


def _integrate(s: State) -> None:
    s.cx += s.vx
    s.cy += s.vy
    s.vx *= VEL_DAMP
    s.vy *= VEL_DAMP


def step(s: State, mx: float, my: float, button: bool) -> State:
    s.ax += ROT_X
    s.ay += ROT_Y
    s.mx = mx
    s.my = my
    s.button = button
    prev_cx, prev_cy = s.cx, s.cy
    hit = math.hypot(mx - s.cx, my - s.cy) <= HIT_RADIUS

    if s.mode == FLOATING and s.grabbed:
        if button:
            s.cx = mx + s.ox
            s.cy = my + s.oy
            s.vx = s.cx - prev_cx
            s.vy = s.cy - prev_cy
        else:
            s.grabbed = False
            s.lock = other_corner(s.lock, mx, my)
            s.mode = SEEKING_LOCK
    elif button and hit:
        s.grabbed = True
        s.ox = s.cx - mx
        s.oy = s.cy - my
        s.mode = FLOATING
    elif hit and s.mode != FLOATING:
        s.mode = FLOATING
        s.grabbed = False
        s.hover = HOVER_FRAMES
        s.lock = other_corner(s.lock, mx, my)
        d = math.hypot(mx - s.cx, my - s.cy)
        if d < 1.0:
            d = 1.0
        s.vx += HOVER_PUSH * (s.cx - mx) / d
        s.vy += HOVER_PUSH * (s.cy - my) / d
        _integrate(s)
    elif s.mode == FLOATING and not s.grabbed:
        _repel(s)
        _integrate(s)
        if s.hover > 0:
            s.hover -= 1
        if s.hover == 0:
            s.mode = SEEKING_LOCK
    else:
        _repel(s)
        _integrate(s)
        tx, ty = corner(s.lock)
        s.cx += (tx - s.cx) * LOCK_LERP
        s.cy += (ty - s.cy) * LOCK_LERP
        dist = math.hypot(s.cx - tx, s.cy - ty)
        if dist < LOCK_EPS:
            s.mode = LOCKED
        elif s.mode == LOCKED:
            s.mode = IDLE_LOCKING

    _clamp(s)
    return s


def rotate_vertex(
    x: float, y: float, z: float, ax: float, ay: float
) -> Tuple[float, float, float]:
    cy, sy = math.cos(ay), math.sin(ay)
    x1 = x * cy + z * sy
    z1 = -x * sy + z * cy
    y1 = y
    cx, sx = math.cos(ax), math.sin(ax)
    x2 = x1
    y2 = y1 * cx - z1 * sx
    z2 = y1 * sx + z1 * cx
    return x2, y2, z2


def project(
    x: float, y: float, z: float, cx: float, cy: float
) -> Tuple[float, float, float]:
    scale = ICO_SCALE * FOCAL / (FOCAL + z)
    return cx + x * scale, cy - y * scale, z


def projected_edges(s: State) -> List[Tuple[float, float, float, float, float]]:
    pts = [rotate_vertex(*VERTS[i], s.ax, s.ay) for i in range(12)]
    out: List[Tuple[float, float, float, float, float]] = []
    for a, b in EDGES:
        ax, ay, az = project(*pts[a], s.cx, s.cy)
        bx, by, bz = project(*pts[b], s.cx, s.cy)
        out.append((ax, ay, bx, by, 0.5 * (az + bz)))
    return out


def mix_wire(z_avg: float) -> int:
    t = max(0.0, min(1.0, (z_avg + 1.0) * 0.5))
    fr, fg, fb = (WIRE_FAR >> 16) & 255, (WIRE_FAR >> 8) & 255, WIRE_FAR & 255
    nr, ng, nb = (WIRE_NEAR >> 16) & 255, (WIRE_NEAR >> 8) & 255, WIRE_NEAR & 255
    r = int(fr + (nr - fr) * t)
    g = int(fg + (ng - fg) * t)
    b = int(fb + (nb - fb) * t)
    return (r << 16) | (g << 8) | b
