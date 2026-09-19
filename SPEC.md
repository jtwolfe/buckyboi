# Icosahedron — shared state machine and formulas

All three ports (Rust, Python, Bend) implement this document. Constants
live in each port under the same names; change them in lockstep.

The **Rust desktop overlay** (`rust/` binary `buckyboi`) is a later UX on
top of the same body. Overlay phases (`VisibleIdle` / `Listening` /
`Hidden`), tap-vs-drag, and the **gaze-follow lock** live in
[UX.md](UX.md). Overlay motion uses `step_overlay_avoid`: same rotate /
repel / lock / drag, but **no hover-push**. The avoid source is
estimated webcam gaze when present, otherwise the pointer. Repulsion is
skipped while that point is on the hit disk so a click or a sustained
look can start Listening. Python and Bend still run the original
800×600 window `step` below.

## Window

| Name | Value | Meaning |
| --- | --- | --- |
| `WIN_W` | `800` | Window width (px) |
| `WIN_H` | `600` | Window height (px) |
| `MARGIN` | `16` | Inset so the wireframe never kisses the edge |
| `BG` | `0x12141A` | Dark background |
| `WIRE_NEAR` | `0xF2F5FC` | Edge color, near camera |
| `WIRE_FAR` | `0x7A8194` | Edge color, far camera |

Target frame time is ~16 ms (60 Hz). Esc (`keycode 27`) or a window
`Close` event quits.

Bend draws a 1024×1024 `Image` quadtree (`depth = 10`) into an 800×600
`Window`; pixels with `x >= 800` or `y >= 600` are unused background.

## Icosahedron

Golden-ratio embedding, then normalize to the unit sphere.

```
PHI = (1 + sqrt(5)) / 2          # 1.618033988749895
N   = sqrt(1 + PHI*PHI)          # 1.902113032590307

# Unnormalized vertices (index 0..11)
( 0, -1, -PHI)  ( 0, -1, +PHI)  ( 0, +1, -PHI)  ( 0, +1, +PHI)
(-1, -PHI,  0)  (-1, +PHI,  0)  (+1, -PHI,  0)  (+1, +PHI,  0)
(-PHI, 0, -1)   (+PHI, 0, -1)   (-PHI, 0, +1)   (+PHI, 0, +1)
```

Unit vertices are those coordinates divided by `N`.

**30 edges** — pairs whose unnormalized distance is `2`:

```
(0,2) (0,4) (0,6) (0,8) (0,9)
(1,3) (1,4) (1,6) (1,10) (1,11)
(2,5) (2,7) (2,8) (2,9)
(3,5) (3,7) (3,10) (3,11)
(4,6) (4,8) (4,10)
(5,7) (5,8) (5,10)
(6,9) (6,11)
(7,9) (7,11)
(8,10) (9,11)
```

| Name | Value | Meaning |
| --- | --- | --- |
| `ICO_SCALE` | `72.0` | Pixel scale after projection |
| `FOCAL` | `4.0` | Perspective: `s = ICO_SCALE * FOCAL / (FOCAL + z)` |
| `ROT_X` | `0.014` | Radians added to `ax` each tick |
| `ROT_Y` | `0.021` | Radians added to `ay` each tick |
| `BOUND_RADIUS` | `104.0` | Conservative projected half-extent used for clamping |
| `LINE_HALF` | `1.1` | Half-width of a wire (px); Bend uses `LINE_HALF²` |

### Rotation then projection

Apply **Ry(`ay`) then Rx(`ax`)** to a unit vertex `(x, y, z)`:

```
# Ry
x1 = x * cos(ay) + z * sin(ay)
z1 = -x * sin(ay) + z * cos(ay)
y1 = y

# Rx
x2 = x1
y2 = y1 * cos(ax) - z1 * sin(ax)
z2 = y1 * sin(ax) + z1 * cos(ax)
```

Then:

```
scale = ICO_SCALE * FOCAL / (FOCAL + z2)
sx    = cx + x2 * scale
sy    = cy - y2 * scale          # model +Y is screen up
```

Screen origin is top-left, +Y down. Edge color interpolates
`WIRE_FAR → WIRE_NEAR` by `t = clamp((z_avg + 1) / 2, 0, 1)` where
`z_avg` is the mean of the two endpoints’ `z2`.

`BOUND_RADIUS` is `ICO_SCALE * FOCAL / (FOCAL - 1) + 8 = 104`, so the
whole wireframe stays inside the clamp box at every orientation.

## Motion constants

| Name | Value | Meaning |
| --- | --- | --- |
| `HIT_RADIUS` | `100.0` | Cursor is “over” the body |
| `REPEL_RADIUS` | `180.0` | Soft-avoid start distance |
| `REPEL_STRENGTH` | `4.2` | Peak per-tick repulsion at d = 0 |
| `LOCK_LERP` | `0.065` | Fraction of remaining lock error applied each tick |
| `LOCK_EPS` | `2.0` | Settle radius (px) |
| `VEL_DAMP` | `0.88` | Velocity multiplier each tick |
| `HOVER_PUSH` | `6.5` | Impulse away from the cursor on hover-grab |
| `HOVER_FRAMES` | `12` | Floating ticks after a hover push |

Lock-box (center must stay here so the wireframe + margin stays on screen):

```
LO_X = MARGIN + BOUND_RADIUS          # 120
HI_X = WIN_W - MARGIN - BOUND_RADIUS  # 680
LO_Y = MARGIN + BOUND_RADIUS          # 120
HI_Y = WIN_H - MARGIN - BOUND_RADIUS  # 480
```

Corners, clockwise from top-left (`lock` index `0..3`):

```
0 TL  (LO_X, LO_Y)   = (120, 120)
1 TR  (HI_X, LO_Y)   = (680, 120)
2 BR  (HI_X, HI_Y)   = (680, 480)
3 BL  (LO_X, HI_Y)   = (120, 480)
```

## Modes

```
IdleLocking   # sliding toward `lock`, not yet inside LOCK_EPS
Locked        # settled on `lock`
Floating      # hover-pushed or mouse-dragged
SeekingLock   # released; sliding toward a *new* `lock`
```

### `other_corner(lock, mx, my) → U32`

Among the three corners whose index `≠ lock`, pick the one **farthest
from the cursor** `(mx, my)`. Ties take the lowest index. This is the
“when disturbed, pick another” rule: the body flees toward the corner
farthest from the mouse.

### Initial state

```
cx, cy     = 400, 300
vx, vy     = 0, 0
ax, ay     = 0, 0
mode       = IdleLocking
lock       = 0                 # home corner = top-left
grabbed    = false
ox, oy     = 0, 0              # grab offset (center - mouse)
hover      = 0
mx, my     = -1000, -1000      # cursor unknown → no repulsion
button     = false
```

The body therefore glides from the center into the top-left lock on
launch, fully on-screen.

## Tick (`step`)

Inputs: previous state, current `(mx, my)`, left-button `button`.

1. **Always rotate:** `ax += ROT_X`, `ay += ROT_Y`.
2. Remember `prev_cx, prev_cy`.
3. `hit = hypot(mx - cx, my - cy) <= HIT_RADIUS`.

Then, exactly one of the following branches (first match wins):

### A. Dragging (`mode == Floating` and `grabbed`)

- If `button` is still down: `cx = mx + ox`, `cy = my + oy`,
  `vx = cx - prev_cx`, `vy = cy - prev_cy`.
- Else (release): `grabbed = false`,
  `lock = other_corner(lock, mx, my)`, `mode = SeekingLock`.
  Keep the release velocity.

### B. Press on the body (`button && hit`)

Enter a drag from any mode:

```
grabbed = true
ox = cx - mx
oy = cy - my
mode = Floating
```

### C. Hover enter (`hit && mode != Floating`)

```
mode    = Floating
grabbed = false
hover   = HOVER_FRAMES
lock    = other_corner(lock, mx, my)
d       = max(hypot(mx - cx, my - cy), 1)
vx     += HOVER_PUSH * (cx - mx) / d
vy     += HOVER_PUSH * (cy - my) / d
```

Then integrate: `cx += vx`, `cy += vy`, `vx *= VEL_DAMP`, `vy *= VEL_DAMP`.

### D. Hover-float (`mode == Floating` and not `grabbed`)

Apply **repulsion** (below), then integrate velocity as in C.
If `hover > 0`, decrement it.
When `hover == 0`, set `mode = SeekingLock` (the lock index was
already retargeted on enter). If the cursor is still over the body
the next tick, branch C fires again and pushes it farther.

### E. Locking (`IdleLocking` / `Locked` / `SeekingLock`)

Apply **repulsion**, integrate velocity, then the lock spring:

```
(tx, ty) = corner(lock)
cx += (tx - cx) * LOCK_LERP
cy += (ty - cy) * LOCK_LERP
```

- If `hypot(cx - tx, cy - ty) < LOCK_EPS`: `mode = Locked`.
- Else if `mode == Locked`: `mode = IdleLocking` (repulsion broke the
  settle; same `lock`).
- `SeekingLock` / `IdleLocking` stay as they are until they settle.

### Repulsion (soft avoid)

Applied in D and E whenever the cursor is not coinciding with the
center and `d = hypot(mx - cx, my - cy)` satisfies `0 < d < REPEL_RADIUS`:

```
t     = 1 - d / REPEL_RADIUS
force = REPEL_STRENGTH * t * t
vx   -= force * (mx - cx) / d
vy   -= force * (my - cy) / d
```

### Clamp (every branch, last)

```
if cx < LO_X: cx = LO_X; vx = 0
if cx > HI_X: cx = HI_X; vx = 0
if cy < LO_Y: cy = LO_Y; vy = 0
if cy > HI_Y: cy = HI_Y; vy = 0
```

This is what “never go off the window edges” means: the **center** is
boxed so the projected wireframe plus `MARGIN` stays inside `[0, WIN_W]
× [0, WIN_H]`.

## Events → inputs

| Event | Effect |
| --- | --- |
| `Move{x, y}` | `mx, my = x, y` |
| `Mouse{x, y, button, down}` | `mx, my = x, y`; any button sets `button = down` |
| `Key{27, True}` | quit |
| `Close` | quit |
| other | ignore |

Ports that poll (Rust `minifb`, Python `pygame`) map the same way:
mouse position each frame, left-button held, Esc / close.

## Headless check

`step` is a pure function of `(state, mx, my, button)`. Each port
exposes it so tests can settle the body at the home corner, retarget
`other_corner`, and assert the clamp box without opening a window.
