# Rotating wireframe icosahedron — Bend 2 port.
# Same simulation as rust/src/sim.rs and python/sim.py. See ../SPEC.md.
#
# Native window (X11 on Linux):
#   bend bend/main.bend -o icosahedron && ./icosahedron
# The JS backend has no Window. `bend PROOF.bend` type-checks without opening one.
import Base

# --- types -----------------------------------------------------------------

type Mode is Data:
  IdleLocking{}
  Locked{}
  Floating{}
  SeekingLock{}

type World is Data:
  World{
    cx: F32, cy: F32, vx: F32, vy: F32, ax: F32, ay: F32, mode: Mode,
    lock: U32, grabbed: Bool, ox: F32, oy: F32, hover: U32, mx: F32, my: F32,
    button: Bool
  }

type Vtx is Data:
  Vtx{x: F32, y: F32, z: F32}

type Seg is Data:
  Seg{ax: F32, ay: F32, bx: F32, by: F32, z: F32}

# --- small picks -----------------------------------------------------------

def Ico.pick_f(c: Bool, a: F32, b: F32) -> F32:
  match c:
    case True{}:
      a
    case False{}:
      b

def Ico.pick_u(c: Bool, a: U32, b: U32) -> U32:
  match c:
    case True{}:
      a
    case False{}:
      b

def Ico.pick_b(c: Bool, a: Bool, b: Bool) -> Bool:
  match c:
    case True{}:
      a
    case False{}:
      b

def Ico.pick_m(c: Bool, a: Mode, b: Mode) -> Mode:
  match c:
    case True{}:
      a
    case False{}:
      b

def Ico.pick_v(c: Bool, a: Vtx, b: Vtx) -> Vtx:
  match c:
    case True{}:
      a
    case False{}:
      b

# --- constants (lockstep with SPEC.md) -------------------------------------

def Ico.lo_x() -> F32:
  120.0

def Ico.hi_x() -> F32:
  680.0

def Ico.lo_y() -> F32:
  120.0

def Ico.hi_y() -> F32:
  480.0

def Ico.bg() -> U32:
  1183770

# --- corners / other_corner ------------------------------------------------

def Ico.corner_x(+lock: U32) -> F32:
  Ico.pick_f(Bool.or(U32.is_eq(lock, 0), U32.is_eq(lock, 3)), Ico.lo_x(),
    Ico.hi_x())

def Ico.corner_y(+lock: U32) -> F32:
  Ico.pick_f(Bool.or(U32.is_eq(lock, 0), U32.is_eq(lock, 1)), Ico.lo_y(),
    Ico.hi_y())

def Ico.score(+i: U32, +lock: U32, +mx: F32, +my: F32) -> F32:
  +tx = Ico.corner_x(i)
  +ty = Ico.corner_y(i)
  Ico.pick_f(U32.is_eq(i, lock), (0.0 - 1.0 : F32),
    ((tx - mx) * (tx - mx) + (ty - my) * (ty - my) : F32))

def Ico.better_u(+i: U32, +lock: U32, +mx: F32, +my: F32, +best: U32, +best_d2: F32) ->
  U32:
  +d2 = Ico.score(i, lock, mx, my)
  Ico.pick_u(F32.is_gt(d2, best_d2), i, best)

def Ico.better_d(+i: U32, +lock: U32, +mx: F32, +my: F32, +best: U32, +best_d2: F32) ->
  F32:
  +d2 = Ico.score(i, lock, mx, my)
  Ico.pick_f(F32.is_gt(d2, best_d2), d2, best_d2)

def Ico.other_corner(+lock: U32, +mx: F32, +my: F32) -> U32:
  +b1 = Ico.better_u(0, lock, mx, my, 0, (0.0 - 1.0 : F32))
  +d1 = Ico.better_d(0, lock, mx, my, 0, (0.0 - 1.0 : F32))
  +b2 = Ico.better_u(1, lock, mx, my, b1, d1)
  +d2 = Ico.better_d(1, lock, mx, my, b1, d1)
  +b3 = Ico.better_u(2, lock, mx, my, b2, d2)
  +d3 = Ico.better_d(2, lock, mx, my, b2, d2)
  Ico.better_u(3, lock, mx, my, b3, d3)

# --- clamp / repel / integrate --------------------------------------------

def Ico.set_x(s: World, cx: F32, vx: F32) -> World:
  World{ox0, cy, vx0, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    button} = s
  World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    button}

def Ico.set_y(s: World, cy: F32, vy: F32) -> World:
  World{cx, oy0, vx, vy0, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    button} = s
  World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    button}

def Ico.clamp_x_hi(high: Bool, s: World) -> World:
  match high:
    case True{}:
      Ico.set_x(s, Ico.hi_x(), 0.0)
    case False{}:
      s

def Ico.clamp_x_lo(low: Bool, s: World) -> World:
  match low:
    case True{}:
      Ico.set_x(s, Ico.lo_x(), 0.0)
    case False{}:
      World{+cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
        button} = s
      Ico.clamp_x_hi(F32.is_gt(cx, Ico.hi_x()),
        World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx,
          my, button})

def Ico.clamp_x(s: World) -> World:
  World{+cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    button} = s
  Ico.clamp_x_lo(F32.is_lt(cx, Ico.lo_x()),
    World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
      button})

def Ico.clamp_y_hi(high: Bool, s: World) -> World:
  match high:
    case True{}:
      Ico.set_y(s, Ico.hi_y(), 0.0)
    case False{}:
      s

def Ico.clamp_y_lo(low: Bool, s: World) -> World:
  match low:
    case True{}:
      Ico.set_y(s, Ico.lo_y(), 0.0)
    case False{}:
      World{cx, +cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
        button} = s
      Ico.clamp_y_hi(F32.is_gt(cy, Ico.hi_y()),
        World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx,
          my, button})

def Ico.clamp_y(s: World) -> World:
  World{cx, +cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    button} = s
  Ico.clamp_y_lo(F32.is_lt(cy, Ico.lo_y()),
    World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
      button})

def Ico.clamp(s: World) -> World:
  Ico.clamp_y(Ico.clamp_x(s))

def Ico.repel_on(on: Bool, +d: F32, +dx: F32, +dy: F32, s: World) -> World:
  match on:
    case False{}:
      s
    case True{}:
      World{cx, cy, +vx, +vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx,
        my, button} = s
      +t = (1.0 - d / 180.0 : F32)
      +force = (4.2 * t * t : F32)
      World{cx, cy, (vx - force * dx / d : F32), (vy - force * dy / d : F32), ax,
        ay, mode, lock, grabbed, ox, oy, hover, mx, my, button}

def Ico.repel_from(+d: F32, +dx: F32, +dy: F32, s: World) -> World:
  Ico.repel_on(Bool.and(F32.is_gt(d, 0.0), F32.is_lt(d, 180.0)), d, dx, dy, s)

def Ico.repel(s: World) -> World:
  World{+cx, +cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, +mx, +my,
    button} = s
  Ico.repel_from(F32.hypot((mx - cx : F32), (my - cy : F32)), (mx - cx : F32),
    (my - cy : F32),
    World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
      button})

def Ico.integrate(s: World) -> World:
  World{+cx, +cy, +vx, +vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    button} = s
  World{(cx + vx : F32), (cy + vy : F32), (vx * 0.88 : F32), (vy * 0.88 : F32),
    ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my, button}

# --- branches of step ------------------------------------------------------

def Ico.is_float(m: Mode) -> Bool:
  match m:
    case Floating{}:
      True{}
    case _:
      False{}

def Ico.was_locked(m: Mode) -> Bool:
  match m:
    case Locked{}:
      True{}
    case _:
      False{}

def Ico.hit(+cx: F32, +cy: F32, +mx: F32, +my: F32) -> Bool:
  F32.is_le(F32.hypot((mx - cx : F32), (my - cy : F32)), 100.0)

def Ico.dec_hover(+h: U32) -> U32:
  Ico.pick_u(U32.is_eq(h, 0), 0, (h - 1 : U32))

def Ico.drag(held: Bool, s: World) -> World:
  match held:
    case True{}:
      World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, +ox, +oy, hover, +mx,
        +my, button} = s
      +cx2 = (mx + ox : F32)
      +cy2 = (my + oy : F32)
      Ico.clamp(World{cx2, cy2, (cx2 - cx : F32), (cy2 - cy : F32), ax, ay,
        mode, lock, grabbed, ox, oy, hover, mx, my, button})
    case False{}:
      World{cx, cy, vx, vy, ax, ay, mode, +lock, grabbed, ox, oy, hover, +mx,
        +my, button} = s
      Ico.clamp(World{cx, cy, vx, vy, ax, ay, SeekingLock{},
        Ico.other_corner(lock, mx, my), False{}, ox, oy, hover, mx, my, button})

def Ico.grab(s: World) -> World:
  World{+cx, +cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, +mx, +my,
    button} = s
  Ico.clamp(World{cx, cy, vx, vy, ax, ay, Floating{}, lock, True{},
    (cx - mx : F32), (cy - my : F32), hover, mx, my, button})

def Ico.hover_enter(s: World) -> World:
  World{+cx, +cy, vx, vy, ax, ay, mode, +lock, grabbed, ox, oy, hover, +mx, +my,
    button} = s
  +d0 = F32.hypot((mx - cx : F32), (my - cy : F32))
  +d = Ico.pick_f(F32.is_lt(d0, 1.0), 1.0, d0)
  Ico.clamp(Ico.integrate(World{cx, cy, (vx + 6.5 * (cx - mx) / d : F32),
    (vy + 6.5 * (cy - my) / d : F32), ax, ay, Floating{},
    Ico.other_corner(lock, mx, my), False{}, ox, oy, 12, mx, my, button}))

def Ico.hover_float(s: World) -> World:
  World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, +hover, mx, my,
    button} = s
  +h2 = Ico.dec_hover(hover)
  Ico.clamp(Ico.integrate(Ico.repel(World{cx, cy, vx, vy, ax, ay,
    Ico.pick_m(U32.is_eq(h2, 0), SeekingLock{}, mode), lock, grabbed, ox, oy,
    h2, mx, my, button})))

def Ico.lock_spring(s: World) -> World:
  World{+cx, +cy, vx, vy, ax, ay, +mode, +lock, grabbed, ox, oy, hover, mx, my,
    button} = s
  +tx = Ico.corner_x(lock)
  +ty = Ico.corner_y(lock)
  +cx3 = (cx + (tx - cx) * 0.065 : F32)
  +cy3 = (cy + (ty - cy) * 0.065 : F32)
  +arrived = F32.is_lt(F32.hypot((cx3 - tx : F32), (cy3 - ty : F32)), 2.0)
  Ico.clamp(World{cx3, cy3, vx, vy, ax, ay,
    Ico.pick_m(arrived, Locked{},
      Ico.pick_m(Ico.was_locked(mode), IdleLocking{}, mode)), lock, grabbed, ox,
    oy, hover, mx, my, button})

def Ico.lock_move(s: World) -> World:
  Ico.lock_spring(Ico.integrate(Ico.repel(s)))

def Ico.branch4(floating: Bool, grabbed: Bool, s: World) -> World:
  match floating grabbed:
    case True{} False{}:
      Ico.hover_float(s)
    case _ _:
      Ico.lock_move(s)

def Ico.branch3(hit: Bool, floating: Bool, grabbed: Bool, s: World) -> World:
  match hit floating:
    case True{} False{}:
      Ico.hover_enter(s)
    case _ _:
      Ico.branch4(floating, grabbed, s)

def Ico.branch2(button: Bool, hit: Bool, floating: Bool, grabbed: Bool, s: World) ->
  World:
  match button hit:
    case True{} True{}:
      Ico.grab(s)
    case _ _:
      Ico.branch3(hit, floating, grabbed, s)

def Ico.branch(floating: Bool, grabbed: Bool, button: Bool, hit: Bool, s: World) ->
  World:
  match floating grabbed:
    case True{} True{}:
      Ico.drag(button, s)
    case _ _:
      Ico.branch2(button, hit, floating, grabbed, s)

def Ico.step(s: World, +mx: F32, +my: F32, +button: Bool) -> World:
  World{+cx, +cy, vx, vy, ax, ay, +mode, lock, +grabbed, ox, oy, hover, mx0,
    my0, button0} = s
  +ax2 = (ax + 0.014 : F32)
  +ay2 = (ay + 0.021 : F32)
  +hit = Ico.hit(cx, cy, mx, my)
  Ico.branch(Ico.is_float(mode), grabbed, button, hit,
    World{cx, cy, vx, vy, ax2, ay2, mode, lock, grabbed, ox, oy, hover, mx, my,
      button})

def Ico.init() -> World:
  World{400.0, 300.0, 0.0, 0.0, 0.0, 0.0, IdleLocking{}, 0, False{}, 0.0, 0.0,
    0, (0.0 - 1000.0 : F32), (0.0 - 1000.0 : F32), False{}}

def Ico.advance(s: World) -> World:
  World{+cx, +cy, +vx, +vy, +ax, +ay, +mode, +lock, +grabbed, +ox, +oy, +hover,
    +mx, +my, +button} = s
  Ico.step(World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx,
    my, button}, mx, my, button)

# --- events ----------------------------------------------------------------

def Ico.with_mouse(s: World, x: U32, y: U32, down: Bool) -> World:
  World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    button} = s
  World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover,
    U32.to_f32(x), U32.to_f32(y), down}

def Ico.with_move(s: World, x: U32, y: U32) -> World:
  World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    +button} = s
  World{cx, cy, vx, vy, ax, ay, mode, lock, grabbed, ox, oy, hover,
    U32.to_f32(x), U32.to_f32(y), button}

def Ico.event(e: Event, s: World) -> Maybe<&1, World>:
  match e:
    case Key{27, True{}}:
      None{}
    case Close{}:
      None{}
    case Mouse{x, y, button, down}:
      Some{Ico.with_mouse(s, x, y, down)}
    case Move{x, y}:
      Some{Ico.with_move(s, x, y)}
    case _:
      Some{s}

def Ico.events(events: List<Event>, s: World) -> Maybe<&1, World>:
  match events:
    case []:
      Some{s}
    case e <> rest:
      Maybe.bind(&1, World, World, Ico.event(e, s), s2 => Ico.events(rest, s2))

def Ico.tick(events: List<Event>, s: World) -> IO(Maybe<&1, World>):
  IO.pure(Maybe<&1, World>, Maybe.bind(&1, World, World, Ico.events(events, s),
    s2 => Some{Ico.advance(s2)}))

# --- geometry --------------------------------------------------------------

def Ico.raw_u(+i: U32) -> Vtx:
  Ico.pick_v(U32.is_eq(i, 0),
    Vtx{0.0, (0.0 - 0.52573111 : F32), (0.0 - 0.85065081 : F32)},
    Ico.pick_v(U32.is_eq(i, 1),
      Vtx{0.0, (0.0 - 0.52573111 : F32), 0.85065081},
      Ico.pick_v(U32.is_eq(i, 2),
        Vtx{0.0, 0.52573111, (0.0 - 0.85065081 : F32)},
        Ico.pick_v(U32.is_eq(i, 3), Vtx{0.0, 0.52573111, 0.85065081},
          Ico.pick_v(U32.is_eq(i, 4),
            Vtx{(0.0 - 0.52573111 : F32), (0.0 - 0.85065081 : F32), 0.0},
            Ico.pick_v(U32.is_eq(i, 5),
              Vtx{(0.0 - 0.52573111 : F32), 0.85065081, 0.0},
              Ico.pick_v(U32.is_eq(i, 6),
                Vtx{0.52573111, (0.0 - 0.85065081 : F32), 0.0},
                Ico.pick_v(U32.is_eq(i, 7), Vtx{0.52573111, 0.85065081, 0.0},
                  Ico.pick_v(U32.is_eq(i, 8),
                    Vtx{(0.0 - 0.85065081 : F32), 0.0, (0.0 - 0.52573111 : F32)},
                    Ico.pick_v(U32.is_eq(i, 9),
                      Vtx{0.85065081, 0.0, (0.0 - 0.52573111 : F32)},
                      Ico.pick_v(U32.is_eq(i, 10),
                        Vtx{(0.0 - 0.85065081 : F32), 0.0, 0.52573111},
                        Vtx{0.85065081, 0.0, 0.52573111})))))))))))

def Ico.raw(i: Nat) -> Vtx:
  Ico.raw_u(U32.from_nat(i))

def Ico.rot(+v: Vtx, +ax: F32, +ay: F32) -> Vtx:
  Vtx{x, y, z} = v
  +cy = F32.cos(ay)
  +sy = F32.sin(ay)
  +x1 = (x * cy + z * sy : F32)
  +z1 = ((0.0 - x) * sy + z * cy : F32)
  +cx = F32.cos(ax)
  +sx = F32.sin(ax)
  Vtx{x1, (y * cx - z1 * sx : F32), (y * sx + z1 * cx : F32)}

def Ico.proj(+v: Vtx, +cx: F32, +cy: F32) -> Vtx:
  Vtx{x, y, z} = v
  +scale = (72.0 * 4.0 / (4.0 + z) : F32)
  Vtx{(cx + x * scale : F32), (cy - y * scale : F32), z}

def Ico.one(i: Nat, +ax: F32, +ay: F32, +cx: F32, +cy: F32) -> Vtx:
  Ico.proj(Ico.rot(Ico.raw(i), ax, ay), cx, cy)

def Ico.mesh(+ax: F32, +ay: F32, +cx: F32, +cy: F32) -> List<&2, Vtx>:
  Ico.one(0n, ax, ay, cx, cy) <> Ico.one(1n, ax, ay, cx, cy)
    <> Ico.one(2n, ax, ay, cx, cy) <> Ico.one(3n, ax, ay, cx, cy)
    <> Ico.one(4n, ax, ay, cx, cy) <> Ico.one(5n, ax, ay, cx, cy)
    <> Ico.one(6n, ax, ay, cx, cy) <> Ico.one(7n, ax, ay, cx, cy)
    <> Ico.one(8n, ax, ay, cx, cy) <> Ico.one(9n, ax, ay, cx, cy)
    <> Ico.one(10n, ax, ay, cx, cy) <> Ico.one(11n, ax, ay, cx, cy) <> []

def Ico.nth(vs: List<&2, Vtx>, i: Nat) -> Vtx:
  match vs i:
    case [] _:
      Vtx{0.0, 0.0, 0.0}
    case h <> t 0n:
      h
    case h <> t 1n+j:
      Ico.nth(t, j)

def Ico.seg_of(+va: Vtx, +vb: Vtx) -> Seg:
  Vtx{ax, ay, az} = va
  Vtx{bx, by, bz} = vb
  Seg{ax, ay, bx, by, (0.5 * (az + bz) : F32)}

def Ico.seg(+vs: List<&2, Vtx>, a: Nat, b: Nat) -> Seg:
  +vs2 = vs
  Ico.seg_of(Ico.nth(vs, a), Ico.nth(vs2, b))

def Ico.segs(+vs: List<&2, Vtx>) -> List<&2, Seg>:
  Ico.seg(vs, 0n, 2n) <> Ico.seg(vs, 0n, 4n) <> Ico.seg(vs, 0n, 6n)
    <> Ico.seg(vs, 0n, 8n) <> Ico.seg(vs, 0n, 9n) <> Ico.seg(vs, 1n, 3n)
    <> Ico.seg(vs, 1n, 4n) <> Ico.seg(vs, 1n, 6n) <> Ico.seg(vs, 1n, 10n)
    <> Ico.seg(vs, 1n, 11n) <> Ico.seg(vs, 2n, 5n) <> Ico.seg(vs, 2n, 7n)
    <> Ico.seg(vs, 2n, 8n) <> Ico.seg(vs, 2n, 9n) <> Ico.seg(vs, 3n, 5n)
    <> Ico.seg(vs, 3n, 7n) <> Ico.seg(vs, 3n, 10n) <> Ico.seg(vs, 3n, 11n)
    <> Ico.seg(vs, 4n, 6n) <> Ico.seg(vs, 4n, 8n) <> Ico.seg(vs, 4n, 10n)
    <> Ico.seg(vs, 5n, 7n) <> Ico.seg(vs, 5n, 8n) <> Ico.seg(vs, 5n, 10n)
    <> Ico.seg(vs, 6n, 9n) <> Ico.seg(vs, 6n, 11n) <> Ico.seg(vs, 7n, 9n)
    <> Ico.seg(vs, 7n, 11n) <> Ico.seg(vs, 8n, 10n) <> Ico.seg(vs, 9n, 11n) <> []

# --- raster ----------------------------------------------------------------

def Ico.dist2(+seg: Seg, +px: F32, +py: F32) -> F32:
  Seg{ax, ay, bx, by, z} = seg
  +abx = (bx - ax : F32)
  +aby = (by - ay : F32)
  +apx = (px - ax : F32)
  +apy = (py - ay : F32)
  +ab2 = (abx * abx + aby * aby + 0.0001 : F32)
  +t = F32.clamp(((apx * abx + apy * aby) / ab2 : F32), 0.0, 1.0)
  +qx = (ax + t * abx : F32)
  +qy = (ay + t * aby : F32)
  ((px - qx) * (px - qx) + (py - qy) * (py - qy) : F32)

def Ico.away(+seg: Seg, +x0: F32, +y0: F32, +s: F32) -> Bool:
  Seg{ax, ay, bx, by, z} = seg
  +pad = {1.6 : F32}
  +minx = (F32.min(ax, bx) - pad : F32)
  +maxx = (F32.max(ax, bx) + pad : F32)
  +miny = (F32.min(ay, by) - pad : F32)
  +maxy = (F32.max(ay, by) + pad : F32)
  Bool.or(F32.is_lt((x0 + s : F32), minx),
    Bool.or(F32.is_lt(maxx, x0),
      Bool.or(F32.is_lt((y0 + s : F32), miny), F32.is_lt(maxy, y0))))

def Ico.all_away(segs: List<&2, Seg>, +x0: F32, +y0: F32, +s: F32) -> Bool:
  match segs:
    case []:
      True{}
    case h <> t:
      Bool.and(Ico.away(h, x0, y0, s), Ico.all_away(t, x0, y0, s))

def Ico.mix(+z: F32) -> U32:
  +t = F32.clamp(((z + 1.0) * 0.5 : F32), 0.0, 1.0)
  +r = F32.to_u32((122.0 + 120.0 * t : F32))
  +g = F32.to_u32((129.0 + 116.0 * t : F32))
  +b = F32.to_u32((148.0 + 104.0 * t : F32))
  (r << 16n .|. g << 8n .|. b : U32)

def Ico.best(
  segs: List<&2, Seg>, +px: F32, +py: F32, +acc_d2: F32, +acc_z: F32
) -> U32:
  match segs:
    case []:
      Ico.pick_u(F32.is_lt(acc_d2, 1.21), Ico.mix(acc_z), Ico.bg())
    case Seg{+ax, +ay, +bx, +by, +z} <> t:
      +d2 = Ico.dist2(Seg{ax, ay, bx, by, z}, px, py)
      +take = F32.is_lt(d2, acc_d2)
      Ico.best(t, px, py, Ico.pick_f(take, d2, acc_d2), Ico.pick_f(take, z, acc_z))

def Ico.color_px(+px: F32, +py: F32, segs: List<&2, Seg>) -> U32:
  Ico.best(segs, px, py, 1000000.0, 0.0)

def Ico.color_in(out: Bool, +x: U32, +y: U32, segs: List<&2, Seg>) -> U32:
  match out:
    case True{}:
      Ico.bg()
    case False{}:
      Ico.color_px((U32.to_f32(x) + 0.5 : F32), (U32.to_f32(y) + 0.5 : F32),
        segs)

def Ico.color(+x: U32, +y: U32, segs: List<&2, Seg>) -> U32:
  Ico.color_in(Bool.or(U32.is_ge(x, 800), U32.is_ge(y, 600)), x, y, segs)

def Ico.flat(d: Nat, +x0: U32, +y0: U32, +segs: List<&2, Seg>) -> Bool:
  +s = U32.to_f32(U32.shln(1, d))
  Bool.or(Bool.or(U32.is_ge(x0, 800), U32.is_ge(y0, 600)),
    Ico.all_away(segs, U32.to_f32(x0), U32.to_f32(y0), s))

def Ico.draw(+d: Nat, flat: Bool, +x0: U32, +y0: U32, +segs: List<&2, Seg>) ->
  Image:
  match d:
    case 0n:
      Pix{Ico.color(x0, y0, segs)}
    case 1n+e:
      match flat:
        case True{}:
          Pix{Ico.color(x0, y0, segs)}
        case False{}:
          +h = U32.shln(1, e)
          +x1 = (x0 + h : U32)
          +y1 = (y0 + h : U32)
          tl tr bl br = Ico.draw(e, Ico.flat(e, x0, y0, segs), x0, y0, segs)
            Ico.draw(e, Ico.flat(e, x1, y0, segs), x1, y0, segs)
            Ico.draw(e, Ico.flat(e, x0, y1, segs), x0, y1, segs)
            Ico.draw(e, Ico.flat(e, x1, y1, segs), x1, y1, segs)
          Qua{tl, tr, bl, br}

def Ico.paint(+segs: List<&2, Seg>) -> Image:
  Ico.draw(10n, Ico.flat(10n, 0, 0, segs), 0, 0, segs)

def Ico.frame(s: World) -> Image:
  World{+cx, +cy, vx, vy, +ax, +ay, mode, lock, grabbed, ox, oy, hover, mx, my,
    button} = s
  Ico.paint(Ico.segs(Ico.mesh(ax, ay, cx, cy)))

def Ico.view(s: World) -> World & Image:
  +s2 = s
  (s2, Ico.frame(s2))

def main() -> IO(Unit):
  App.run(~World, ~App{Ico.view, Ico.tick}, "Icosahedron", 800, 600, Ico.init())
