//! Shared icosahedron simulation. Constants and formulas match SPEC.md.

pub const WIN_W: usize = 800;
pub const WIN_H: usize = 600;
pub const MARGIN: f32 = 16.0;
pub const BG: u32 = 0x12141A;
pub const WIRE_NEAR: u32 = 0xF2F5FC;
pub const WIRE_FAR: u32 = 0x7A8194;

pub const PHI: f32 = 1.618_034;
const NORM: f32 = 1.902_113;

pub const ICO_SCALE: f32 = 72.0;
pub const FOCAL: f32 = 4.0;
pub const ROT_X: f32 = 0.014;
pub const ROT_Y: f32 = 0.021;
pub const BOUND_RADIUS: f32 = 104.0;
pub const LINE_HALF: f32 = 1.1;

pub const HIT_RADIUS: f32 = 100.0;
pub const REPEL_RADIUS: f32 = 180.0;
pub const REPEL_STRENGTH: f32 = 4.2;
pub const LOCK_LERP: f32 = 0.065;
pub const LOCK_EPS: f32 = 2.0;
pub const VEL_DAMP: f32 = 0.88;
pub const HOVER_PUSH: f32 = 6.5;
pub const HOVER_FRAMES: u32 = 12;

pub const LO_X: f32 = MARGIN + BOUND_RADIUS; // 120
pub const HI_X: f32 = (WIN_W as f32) - MARGIN - BOUND_RADIUS; // 680
pub const LO_Y: f32 = MARGIN + BOUND_RADIUS; // 120
pub const HI_Y: f32 = (WIN_H as f32) - MARGIN - BOUND_RADIUS; // 480

pub const CORNERS: [(f32, f32); 4] = [(LO_X, LO_Y), (HI_X, LO_Y), (HI_X, HI_Y), (LO_X, HI_Y)];

const RAW: [(f32, f32, f32); 12] = [
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
];

pub const EDGES: [(usize, usize); 30] = [
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
];

pub const IDLE_LOCKING: u8 = 0;
pub const LOCKED: u8 = 1;
pub const FLOATING: u8 = 2;
pub const SEEKING_LOCK: u8 = 3;

#[derive(Clone, Debug)]
pub struct State {
    pub cx: f32,
    pub cy: f32,
    pub vx: f32,
    pub vy: f32,
    pub ax: f32,
    pub ay: f32,
    pub mode: u8,
    pub lock: u8,
    pub grabbed: bool,
    pub ox: f32,
    pub oy: f32,
    pub hover: u32,
    pub mx: f32,
    pub my: f32,
    pub button: bool,
    pub sw: f32,
    pub sh: f32,
}

pub fn verts() -> [(f32, f32, f32); 12] {
    let mut out = [(0.0, 0.0, 0.0); 12];
    for i in 0..12 {
        let (x, y, z) = RAW[i];
        out[i] = (x / NORM, y / NORM, z / NORM);
    }
    out
}

pub fn initial() -> State {
    initial_on(WIN_W as f32, WIN_H as f32)
}

pub fn initial_on(sw: f32, sh: f32) -> State {
    State {
        cx: sw * 0.5,
        cy: sh * 0.5,
        vx: 0.0,
        vy: 0.0,
        ax: 0.0,
        ay: 0.0,
        mode: IDLE_LOCKING,
        lock: 0,
        grabbed: false,
        ox: 0.0,
        oy: 0.0,
        hover: 0,
        mx: -1000.0,
        my: -1000.0,
        button: false,
        sw,
        sh,
    }
}

pub fn lock_box(sw: f32, sh: f32) -> (f32, f32, f32, f32) {
    let lo_x = MARGIN + BOUND_RADIUS;
    let hi_x = sw - MARGIN - BOUND_RADIUS;
    let lo_y = MARGIN + BOUND_RADIUS;
    let hi_y = sh - MARGIN - BOUND_RADIUS;
    (lo_x, hi_x, lo_y, hi_y)
}

pub fn corner(lock: u8) -> (f32, f32) {
    corner_on(lock, WIN_W as f32, WIN_H as f32)
}

pub fn corner_on(lock: u8, sw: f32, sh: f32) -> (f32, f32) {
    let (lo_x, hi_x, lo_y, hi_y) = lock_box(sw, sh);
    match lock & 3 {
        0 => (lo_x, lo_y),
        1 => (hi_x, lo_y),
        2 => (hi_x, hi_y),
        _ => (lo_x, hi_y),
    }
}

/// Farthest corner from the cursor, excluding `lock`. Ties: lowest index.
pub fn other_corner(lock: u8, mx: f32, my: f32) -> u8 {
    other_corner_on(lock, mx, my, WIN_W as f32, WIN_H as f32)
}

pub fn other_corner_on(lock: u8, mx: f32, my: f32, sw: f32, sh: f32) -> u8 {
    let mut best: i32 = -1;
    let mut best_d2 = -1.0_f32;
    for i in 0..4u8 {
        if i == lock {
            continue;
        }
        let (tx, ty) = corner_on(i, sw, sh);
        let d2 = (tx - mx) * (tx - mx) + (ty - my) * (ty - my);
        if d2 > best_d2 {
            best_d2 = d2;
            best = i as i32;
        }
    }
    best as u8
}

fn clamp(s: &mut State) {
    let (lo_x, hi_x, lo_y, hi_y) = lock_box(s.sw, s.sh);
    if s.cx < lo_x {
        s.cx = lo_x;
        s.vx = 0.0;
    }
    if s.cx > hi_x {
        s.cx = hi_x;
        s.vx = 0.0;
    }
    if s.cy < lo_y {
        s.cy = lo_y;
        s.vy = 0.0;
    }
    if s.cy > hi_y {
        s.cy = hi_y;
        s.vy = 0.0;
    }
}

fn repel(s: &mut State) {
    repel_from(s, s.mx, s.my, REPEL_RADIUS, REPEL_STRENGTH);
}

pub fn repel_from(s: &mut State, px: f32, py: f32, radius: f32, strength: f32) {
    let dx = px - s.cx;
    let dy = py - s.cy;
    let d = dx.hypot(dy);
    if d <= 0.0 || d >= radius {
        return;
    }
    let t = 1.0 - d / radius;
    let force = strength * t * t;
    s.vx -= force * dx / d;
    s.vy -= force * dy / d;
}

fn integrate(s: &mut State) {
    s.cx += s.vx;
    s.cy += s.vy;
    s.vx *= VEL_DAMP;
    s.vy *= VEL_DAMP;
}

pub fn step(s: &mut State, mx: f32, my: f32, button: bool) {
    s.ax += ROT_X;
    s.ay += ROT_Y;
    s.mx = mx;
    s.my = my;
    s.button = button;
    let prev_cx = s.cx;
    let prev_cy = s.cy;
    let hit = (mx - s.cx).hypot(my - s.cy) <= HIT_RADIUS;

    if s.mode == FLOATING && s.grabbed {
        if button {
            s.cx = mx + s.ox;
            s.cy = my + s.oy;
            s.vx = s.cx - prev_cx;
            s.vy = s.cy - prev_cy;
        } else {
            s.grabbed = false;
            s.lock = other_corner_on(s.lock, mx, my, s.sw, s.sh);
            s.mode = SEEKING_LOCK;
        }
    } else if button && hit {
        s.grabbed = true;
        s.ox = s.cx - mx;
        s.oy = s.cy - my;
        s.mode = FLOATING;
    } else if hit && s.mode != FLOATING {
        s.mode = FLOATING;
        s.grabbed = false;
        s.hover = HOVER_FRAMES;
        s.lock = other_corner_on(s.lock, mx, my, s.sw, s.sh);
        let mut d = (mx - s.cx).hypot(my - s.cy);
        if d < 1.0 {
            d = 1.0;
        }
        s.vx += HOVER_PUSH * (s.cx - mx) / d;
        s.vy += HOVER_PUSH * (s.cy - my) / d;
        integrate(s);
    } else if s.mode == FLOATING && !s.grabbed {
        repel(s);
        integrate(s);
        if s.hover > 0 {
            s.hover -= 1;
        }
        if s.hover == 0 {
            s.mode = SEEKING_LOCK;
        }
    } else {
        repel(s);
        integrate(s);
        let (tx, ty) = corner_on(s.lock, s.sw, s.sh);
        s.cx += (tx - s.cx) * LOCK_LERP;
        s.cy += (ty - s.cy) * LOCK_LERP;
        let dist = (s.cx - tx).hypot(s.cy - ty);
        if dist < LOCK_EPS {
            s.mode = LOCKED;
        } else if s.mode == LOCKED {
            s.mode = IDLE_LOCKING;
        }
    }

    clamp(s);
}

/// Overlay physics: rotate, soft avoid, corner lock, optional drag.
/// No hover-push — that would steal the click-to-listen gesture.
pub fn step_overlay(s: &mut State, mx: f32, my: f32, dragging: bool) {
    step_overlay_avoid(s, mx, my, dragging, None);
}

/// `gaze` replaces the mouse as the avoid source when present.
/// While the avoid point is on the hit disk the body does not flee.
pub fn step_overlay_avoid(
    s: &mut State,
    mx: f32,
    my: f32,
    dragging: bool,
    gaze: Option<(f32, f32)>,
) {
    s.ax += ROT_X;
    s.ay += ROT_Y;
    s.mx = mx;
    s.my = my;
    let prev_cx = s.cx;
    let prev_cy = s.cy;

    if dragging {
        if !s.grabbed {
            s.grabbed = true;
            s.ox = s.cx - mx;
            s.oy = s.cy - my;
            s.mode = FLOATING;
        }
        s.cx = mx + s.ox;
        s.cy = my + s.oy;
        s.vx = s.cx - prev_cx;
        s.vy = s.cy - prev_cy;
        s.button = true;
        clamp(s);
        return;
    }

    if s.grabbed {
        s.grabbed = false;
        s.lock = other_corner_on(s.lock, mx, my, s.sw, s.sh);
        s.mode = SEEKING_LOCK;
    }
    s.button = false;
    let (ax, ay, radius, strength) = match gaze {
        Some((gx, gy)) => (
            gx,
            gy,
            crate::gaze::GAZE_REPEL_RADIUS,
            crate::gaze::GAZE_REPEL_STRENGTH,
        ),
        None => (mx, my, REPEL_RADIUS, REPEL_STRENGTH),
    };
    if !hit_test(s, ax, ay) {
        repel_from(s, ax, ay, radius, strength);
    }
    integrate(s);
    let (tx, ty) = corner_on(s.lock, s.sw, s.sh);
    s.cx += (tx - s.cx) * LOCK_LERP;
    s.cy += (ty - s.cy) * LOCK_LERP;
    let dist = (s.cx - tx).hypot(s.cy - ty);
    if dist < LOCK_EPS {
        s.mode = LOCKED;
    } else if s.mode == LOCKED {
        s.mode = IDLE_LOCKING;
    }
    clamp(s);
}

pub fn spin(s: &mut State) {
    s.ax += ROT_X;
    s.ay += ROT_Y;
}

pub fn hit_test(s: &State, mx: f32, my: f32) -> bool {
    (mx - s.cx).hypot(my - s.cy) <= HIT_RADIUS
}

pub fn rotate_vertex(x: f32, y: f32, z: f32, ax: f32, ay: f32) -> (f32, f32, f32) {
    let (cy, sy) = (ay.cos(), ay.sin());
    let x1 = x * cy + z * sy;
    let z1 = -x * sy + z * cy;
    let y1 = y;
    let (cx, sx) = (ax.cos(), ax.sin());
    let x2 = x1;
    let y2 = y1 * cx - z1 * sx;
    let z2 = y1 * sx + z1 * cx;
    (x2, y2, z2)
}

pub fn project(x: f32, y: f32, z: f32, cx: f32, cy: f32) -> (f32, f32, f32) {
    let scale = ICO_SCALE * FOCAL / (FOCAL + z);
    (cx + x * scale, cy - y * scale, z)
}

/// (x0, y0, x1, y1, z_avg) for each of the 30 edges.
pub fn projected_edges(s: &State) -> [(f32, f32, f32, f32, f32); 30] {
    let v = verts();
    let mut pts = [(0.0, 0.0, 0.0); 12];
    for i in 0..12 {
        let (x, y, z) = v[i];
        pts[i] = rotate_vertex(x, y, z, s.ax, s.ay);
    }
    let mut out = [(0.0, 0.0, 0.0, 0.0, 0.0); 30];
    for (k, &(a, b)) in EDGES.iter().enumerate() {
        let (ax, ay, az) = project(pts[a].0, pts[a].1, pts[a].2, s.cx, s.cy);
        let (bx, by, bz) = project(pts[b].0, pts[b].1, pts[b].2, s.cx, s.cy);
        out[k] = (ax, ay, bx, by, 0.5 * (az + bz));
    }
    out
}

pub fn mix_wire(z_avg: f32) -> u32 {
    let t = ((z_avg + 1.0) * 0.5).clamp(0.0, 1.0);
    let fr = ((WIRE_FAR >> 16) & 255) as f32;
    let fg = ((WIRE_FAR >> 8) & 255) as f32;
    let fb = (WIRE_FAR & 255) as f32;
    let nr = ((WIRE_NEAR >> 16) & 255) as f32;
    let ng = ((WIRE_NEAR >> 8) & 255) as f32;
    let nb = (WIRE_NEAR & 255) as f32;
    let r = (fr + (nr - fr) * t) as u32;
    let g = (fg + (ng - fg) * t) as u32;
    let b = (fb + (nb - fb) * t) as u32;
    (r << 16) | (g << 8) | b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twelve_unit_vertices() {
        for (x, y, z) in verts() {
            let n = (x * x + y * y + z * z).sqrt();
            assert!((n - 1.0).abs() < 1e-5, "{n}");
        }
    }

    #[test]
    fn thirty_edges() {
        assert_eq!(EDGES.len(), 30);
    }

    #[test]
    fn other_corner_never_same() {
        for lock in 0..4u8 {
            for (mx, my) in [(0.0, 0.0), (800.0, 0.0), (800.0, 600.0), (0.0, 600.0)] {
                let got = other_corner(lock, mx, my);
                assert_ne!(got, lock);
                assert!(got < 4);
            }
        }
        assert_eq!(other_corner(0, 0.0, 0.0), 2);
        assert_eq!(other_corner(2, 800.0, 600.0), 0);
    }

    #[test]
    fn idles_into_home_corner() {
        let mut s = initial();
        for _ in 0..400 {
            step(&mut s, -1000.0, -1000.0, false);
        }
        let (tx, ty) = corner(0);
        assert_eq!(s.mode, LOCKED);
        assert!((s.cx - tx).hypot(s.cy - ty) < 2.5);
        assert!(s.cx >= LO_X && s.cx <= HI_X);
        assert!(s.cy >= LO_Y && s.cy <= HI_Y);
    }

    #[test]
    fn clamp_never_leaves_box() {
        let mut s = initial();
        s.cx = 10.0;
        s.cy = 10.0;
        s.vx = -40.0;
        s.vy = -40.0;
        step(&mut s, -1000.0, -1000.0, false);
        assert!(s.cx >= LO_X && s.cx <= HI_X);
        assert!(s.cy >= LO_Y && s.cy <= HI_Y);
    }

    #[test]
    fn drag_follows_then_releases() {
        let mut s = initial();
        for _ in 0..400 {
            step(&mut s, -1000.0, -1000.0, false);
        }
        let (ox, oy) = (s.cx, s.cy);
        step(&mut s, ox, oy, true);
        assert!(s.grabbed);
        step(&mut s, 400.0, 300.0, true);
        assert!((s.cx - 400.0).abs() < 1.0);
        assert!((s.cy - 300.0).abs() < 1.0);
        let prev = s.lock;
        step(&mut s, 400.0, 300.0, false);
        assert!(!s.grabbed);
        assert_eq!(s.mode, SEEKING_LOCK);
        assert_ne!(s.lock, prev);
    }

    #[test]
    fn always_rotates() {
        let mut s = initial();
        step(&mut s, -1000.0, -1000.0, false);
        assert!(s.ax > 0.0 && s.ay > 0.0);
    }

    #[test]
    fn projected_edge_count() {
        let s = initial();
        assert_eq!(projected_edges(&s).len(), 30);
        let sorted = crate::draw::edges_far_to_near(&s);
        assert_eq!(sorted.len(), 30);
        for i in 1..sorted.len() {
            assert!(sorted[i].4 + 1e-4 >= sorted[i - 1].4);
        }
    }

    #[test]
    fn lock_box_keeps_wireframe_on_screen() {
        assert!(LO_X - BOUND_RADIUS >= MARGIN);
        assert!(HI_X + BOUND_RADIUS <= WIN_W as f32 - MARGIN);
        assert!(LO_Y - BOUND_RADIUS >= MARGIN);
        assert!(HI_Y + BOUND_RADIUS <= WIN_H as f32 - MARGIN);
    }

    #[test]
    fn overlay_hover_does_not_grab() {
        let mut s = initial();
        for _ in 0..400 {
            step_overlay(&mut s, -1000.0, -1000.0, false);
        }
        assert_eq!(s.mode, LOCKED);
        let (cx, cy) = (s.cx, s.cy);
        step_overlay(&mut s, cx, cy, false);
        assert!(!s.grabbed);
        assert_ne!(s.mode, FLOATING);
    }

    #[test]
    fn overlay_drag_follows() {
        let mut s = initial();
        for _ in 0..400 {
            step_overlay(&mut s, -1000.0, -1000.0, false);
        }
        let (ox, oy) = (s.cx, s.cy);
        step_overlay(&mut s, ox, oy, true);
        assert!(s.grabbed);
        step_overlay(&mut s, 400.0, 300.0, true);
        assert!((s.cx - 400.0).abs() < 1.0);
        assert!((s.cy - 300.0).abs() < 1.0);
    }

    #[test]
    fn overlay_over_hit_does_not_flee() {
        let mut s = initial();
        for _ in 0..400 {
            step_overlay(&mut s, -1000.0, -1000.0, false);
        }
        let (cx, cy) = (s.cx, s.cy);
        for _ in 0..12 {
            step_overlay(&mut s, cx, cy, false);
        }
        assert!((s.cx - cx).abs() < 1.0);
        assert!((s.cy - cy).abs() < 1.0);
        assert!(!s.grabbed);
    }

    #[test]
    fn overlay_gaze_pushes_away() {
        let mut s = initial();
        for _ in 0..400 {
            step_overlay(&mut s, -1000.0, -1000.0, false);
        }
        let (cx, cy) = (s.cx, s.cy);
        // Outside the hit disk, inside the gaze-repel radius; flee right
        // (the home corner's left wall would eat a leftward push).
        for _ in 0..30 {
            step_overlay_avoid(&mut s, -1000.0, -1000.0, false, Some((cx - 140.0, cy)));
        }
        assert!(s.cx > cx + 1.5, "expected flee right, cx {cx} → {}", s.cx);
    }

    #[test]
    fn overlay_gaze_on_body_does_not_flee() {
        let mut s = initial();
        for _ in 0..400 {
            step_overlay(&mut s, -1000.0, -1000.0, false);
        }
        let (cx, cy) = (s.cx, s.cy);
        for _ in 0..12 {
            step_overlay_avoid(&mut s, -1000.0, -1000.0, false, Some((cx, cy)));
        }
        assert!((s.cx - cx).abs() < 1.0);
        assert!((s.cy - cy).abs() < 1.0);
    }
}
