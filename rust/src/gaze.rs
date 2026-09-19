//! Approximate on-screen gaze from a face (or a simulated proxy).
//!
//! This is a prototype: face position in the camera frame is mapped to a
//! screen look-target, then smoothed. It is **not** research-grade eye tracking.
//! Camera I/O lives in `camera.rs` (feature `gaze`). This module is pure.

use crate::ux::{LISTEN_MAX_MS, LISTEN_MIN_MS, UxEvent};

pub const GAZE_SMOOTH: f32 = 0.14;
pub const GAZE_HOLD_MS: u64 = 280;
pub const GAZE_GRACE_MS: u64 = 220;
pub const GAZE_LOCK_MIN_MS: u32 = 5_000;
pub const GAZE_LOCK_MAX_MS: u32 = 15_000;
pub const GAZE_ENTER_SLACK: f32 = 8.0;
pub const GAZE_LEAVE_SLACK: f32 = 22.0;
pub const GAZE_REPEL_RADIUS: f32 = 230.0;
pub const GAZE_REPEL_STRENGTH: f32 = 1.55;
pub const FACE_GAIN: f32 = 2.15;

/// Deterministic 5–15 s gaze-follow window (same span as listen).
pub fn gaze_lock_ms(seed: u64) -> u32 {
    let span = (GAZE_LOCK_MAX_MS - GAZE_LOCK_MIN_MS) as u64;
    GAZE_LOCK_MIN_MS + (seed % (span + 1)) as u32
}

/// Map a face (or iris) center in a camera frame to a screen-space look target.
/// `mirror` is true for a typical user-facing webcam (selfie / mirror view).
pub fn face_to_screen(
    fx: f32,
    fy: f32,
    frame_w: f32,
    frame_h: f32,
    screen_w: f32,
    screen_h: f32,
    mirror: bool,
) -> (f32, f32) {
    if frame_w <= 1.0 || frame_h <= 1.0 {
        return (screen_w * 0.5, screen_h * 0.5);
    }
    let mut nx = (fx / frame_w).clamp(0.0, 1.0);
    let ny = (fy / frame_h).clamp(0.0, 1.0);
    if mirror {
        nx = 1.0 - nx;
    }
    let sx = (0.5 + (nx - 0.5) * FACE_GAIN).clamp(0.03, 0.97) * screen_w;
    let sy = (0.5 + (ny - 0.5) * FACE_GAIN).clamp(0.03, 0.97) * screen_h;
    (sx, sy)
}

#[derive(Clone, Debug)]
pub struct GazeSmoother {
    pub x: f32,
    pub y: f32,
    pub ready: bool,
    pub last_ms: u64,
    pub alpha: f32,
}

impl GazeSmoother {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            ready: false,
            last_ms: 0,
            alpha: GAZE_SMOOTH,
        }
    }

    pub fn push(&mut self, x: f32, y: f32, now_ms: u64) -> (f32, f32) {
        if !self.ready {
            self.x = x;
            self.y = y;
            self.ready = true;
        } else {
            self.x += (x - self.x) * self.alpha;
            self.y += (y - self.y) * self.alpha;
        }
        self.last_ms = now_ms;
        (self.x, self.y)
    }

    /// Latest smoothed point if a sample arrived recently.
    pub fn current(&self, now_ms: u64) -> Option<(f32, f32)> {
        if self.ready && now_ms.saturating_sub(self.last_ms) <= GAZE_HOLD_MS {
            Some((self.x, self.y))
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.ready = false;
    }
}

impl Default for GazeSmoother {
    fn default() -> Self {
        Self::new()
    }
}

/// Continuous gaze-on-buddy timer. Fires `StartListen` after a random 5–15 s.
#[derive(Clone, Debug)]
pub struct GazeLock {
    pub since_ms: Option<u64>,
    pub lost_ms: Option<u64>,
    pub need_ms: u32,
    pub on: bool,
}

impl GazeLock {
    pub fn new(seed: u64) -> Self {
        Self {
            since_ms: None,
            lost_ms: None,
            need_ms: gaze_lock_ms(seed),
            on: false,
        }
    }

    pub fn with_need(need_ms: u32) -> Self {
        Self {
            since_ms: None,
            lost_ms: None,
            need_ms: need_ms.max(1),
            on: false,
        }
    }

    pub fn reset(&mut self) {
        self.since_ms = None;
        self.lost_ms = None;
        self.on = false;
    }

    pub fn progress(&self, now_ms: u64) -> f32 {
        match self.since_ms {
            Some(t) if self.need_ms > 0 => {
                ((now_ms.saturating_sub(t) as f32) / self.need_ms as f32).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }

    /// `over` should already include enter/leave hysteresis.
    pub fn update(&mut self, now_ms: u64, over: bool) -> UxEvent {
        if over {
            self.lost_ms = None;
            self.on = true;
            if self.since_ms.is_none() {
                self.since_ms = Some(now_ms);
            }
            if now_ms.saturating_sub(self.since_ms.unwrap()) >= self.need_ms as u64 {
                return UxEvent::StartListen;
            }
            return UxEvent::None;
        }
        if self.since_ms.is_none() {
            self.on = false;
            return UxEvent::None;
        }
        if self.lost_ms.is_none() {
            self.lost_ms = Some(now_ms);
        }
        if now_ms.saturating_sub(self.lost_ms.unwrap()) >= GAZE_GRACE_MS {
            self.reset();
        }
        UxEvent::None
    }
}

/// Hysteresis: easier to stay on the buddy than to first acquire it.
pub fn gaze_over_hysteresis(was_on: bool, dist: f32, hit_radius: f32) -> bool {
    if was_on {
        dist <= hit_radius + GAZE_LEAVE_SLACK
    } else {
        dist <= hit_radius + GAZE_ENTER_SLACK
    }
}

/// Scripted look-target for demos with no camera: approaches the buddy, then holds.
pub fn chase_gaze(
    now_ms: u64,
    origin_ms: u64,
    cx: f32,
    cy: f32,
    screen_w: f32,
    screen_h: f32,
) -> (f32, f32) {
    let t = now_ms.saturating_sub(origin_ms) as f32 / 1000.0;
    let away_x = (screen_w - cx).clamp(80.0, screen_w - 80.0);
    let away_y = (screen_h - cy).clamp(80.0, screen_h - 80.0);
    let blend = ((t - 0.8) / 2.2).clamp(0.0, 1.0);
    let blend = blend * blend * (3.0 - 2.0 * blend);
    (
        away_x + (cx - away_x) * blend,
        away_y + (cy - away_y) * blend,
    )
}

/// Packed RGB8 skin + dark-iris proxy. Returns face-center (and iris if found)
/// in frame pixels. Used by the camera thread and unit tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceGuess {
    pub cx: f32,
    pub cy: f32,
    pub iris_x: Option<f32>,
    pub iris_y: Option<f32>,
}

impl FaceGuess {
    pub fn look_point(&self) -> (f32, f32) {
        match (self.iris_x, self.iris_y) {
            (Some(x), Some(y)) => (x, y),
            _ => (self.cx, self.cy),
        }
    }
}

fn is_skin(r: u8, g: u8, b: u8) -> bool {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;
    let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let cb = 128.0 - 0.168736 * rf - 0.331264 * gf + 0.5 * bf;
    let cr = 128.0 + 0.5 * rf - 0.418688 * gf - 0.081312 * bf;
    (77.0..128.0).contains(&cb) && (133.0..174.0).contains(&cr) && y > 38.0 && r > g && r > 60
}

/// Largest skin-colored blob → face center; darkest pixels in the upper half → iris.
pub fn estimate_face_rgb(rgb: &[u8], w: u32, h: u32) -> Option<FaceGuess> {
    if w < 8 || h < 8 {
        return None;
    }
    let w = w as usize;
    let h = h as usize;
    let expect = w * h * 3;
    if rgb.len() < expect {
        return None;
    }
    let mut mask = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            mask[y * w + x] = is_skin(rgb[i], rgb[i + 1], rgb[i + 2]);
        }
    }
    let mut seen = vec![false; w * h];
    let mut best_area = 0usize;
    let mut best_cx = 0.0f32;
    let mut best_cy = 0.0f32;
    let mut best_min = (0usize, 0usize);
    let mut best_max = (0usize, 0usize);
    let mut stack = Vec::new();
    for y0 in 0..h {
        for x0 in 0..w {
            let start = y0 * w + x0;
            if !mask[start] || seen[start] {
                continue;
            }
            stack.clear();
            stack.push((x0, y0));
            seen[start] = true;
            let mut area = 0usize;
            let mut sx = 0u64;
            let mut sy = 0u64;
            let mut min_x = x0;
            let mut max_x = x0;
            let mut min_y = y0;
            let mut max_y = y0;
            while let Some((x, y)) = stack.pop() {
                area += 1;
                sx += x as u64;
                sy += y as u64;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                for (nx, ny) in [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ] {
                    if nx >= w || ny >= h {
                        continue;
                    }
                    let j = ny * w + nx;
                    if mask[j] && !seen[j] {
                        seen[j] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            if area > best_area {
                best_area = area;
                best_cx = sx as f32 / area as f32;
                best_cy = sy as f32 / area as f32;
                best_min = (min_x, min_y);
                best_max = (max_x, max_y);
            }
        }
    }
    let min_area = ((w * h) / 80).max(24);
    if best_area < min_area {
        return None;
    }
    let (x0, y0) = best_min;
    let (x1, y1) = best_max;
    let mid_y = y0 + (y1.saturating_sub(y0) * 45 / 100);
    let mut dark_s = 0u64;
    let mut dark_n = 0u64;
    let mut iris_x = 0.0f32;
    let mut iris_y = 0.0f32;
    for y in y0..=mid_y.min(h.saturating_sub(1)) {
        for x in x0..=x1.min(w.saturating_sub(1)) {
            let i = (y * w + x) * 3;
            let yv = (rgb[i] as u32 + rgb[i + 1] as u32 + rgb[i + 2] as u32) / 3;
            if yv < 70 {
                dark_n += 1;
                dark_s += yv as u64;
                iris_x += x as f32;
                iris_y += y as f32;
            }
        }
    }
    let iris = if dark_n > 4 {
        Some((iris_x / dark_n as f32, iris_y / dark_n as f32))
    } else {
        let _ = dark_s;
        None
    };
    Some(FaceGuess {
        cx: best_cx,
        cy: best_cy,
        iris_x: iris.map(|p| p.0),
        iris_y: iris.map(|p| p.1),
    })
}

/// Convert YUYV 4:2:2 packed bytes to RGB8.
pub fn yuyv_to_rgb(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let mut out = vec![0u8; w * h * 3];
    let mut si = 0usize;
    for y in 0..h {
        let mut x = 0usize;
        while x + 1 < w && si + 3 < src.len() {
            let y0 = src[si] as i32;
            let u = src[si + 1] as i32 - 128;
            let y1 = src[si + 2] as i32;
            let v = src[si + 3] as i32 - 128;
            let p0 = (y * w + x) * 3;
            let p1 = (y * w + x + 1) * 3;
            yuv_pixel(&mut out, p0, y0, u, v);
            yuv_pixel(&mut out, p1, y1, u, v);
            si += 4;
            x += 2;
        }
    }
    out
}

fn yuv_pixel(out: &mut [u8], i: usize, y: i32, u: i32, v: i32) {
    let r = (y + (359 * v) / 256).clamp(0, 255) as u8;
    let g = (y - (88 * u) / 256 - (183 * v) / 256).clamp(0, 255) as u8;
    let b = (y + (454 * u) / 256).clamp(0, 255) as u8;
    out[i] = r;
    out[i + 1] = g;
    out[i + 2] = b;
}

// Re-export listen span so env docs stay consistent.
#[allow(dead_code)]
const _LISTEN_SPAN_CHECK: u32 = LISTEN_MAX_MS - LISTEN_MIN_MS;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_duration_in_range() {
        for seed in [0u64, 1, 17, 999, 1_000_000] {
            let d = gaze_lock_ms(seed);
            assert!((GAZE_LOCK_MIN_MS..=GAZE_LOCK_MAX_MS).contains(&d), "{d}");
        }
        assert_eq!(gaze_lock_ms(0), GAZE_LOCK_MIN_MS);
        assert_eq!(gaze_lock_ms(10_000), GAZE_LOCK_MAX_MS);
    }

    #[test]
    fn smoother_eases_toward_sample() {
        let mut s = GazeSmoother::new();
        s.alpha = 0.25;
        let (x, y) = s.push(100.0, 50.0, 1);
        assert!((x - 100.0).abs() < 0.01 && (y - 50.0).abs() < 0.01);
        let (x, y) = s.push(200.0, 50.0, 2);
        assert!(x > 100.0 && x < 200.0);
        assert!((y - 50.0).abs() < 0.01);
        assert!(s.current(2 + GAZE_HOLD_MS).is_some());
        assert!(s.current(2 + GAZE_HOLD_MS + 1).is_none());
    }

    #[test]
    fn face_to_screen_mirrors_and_gains() {
        let (sx, sy) = face_to_screen(0.0, 120.0, 320.0, 240.0, 1920.0, 1080.0, true);
        assert!(sx > 1920.0 * 0.7, "{sx}");
        assert!((sy - 540.0).abs() < 1.0, "{sy}");
        let (cx, cy) = face_to_screen(160.0, 120.0, 320.0, 240.0, 1920.0, 1080.0, true);
        assert!((cx - 960.0).abs() < 1.0 && (cy - 540.0).abs() < 1.0);
    }

    #[test]
    fn gaze_lock_needs_continuous_dwell() {
        let mut lock = GazeLock::with_need(5_000);
        assert_eq!(lock.update(1000, false), UxEvent::None);
        assert_eq!(lock.update(1000, true), UxEvent::None);
        assert!((lock.progress(3500) - 0.5).abs() < 0.02);
        assert_eq!(lock.update(3500, true), UxEvent::None);
        assert_eq!(lock.update(6000, true), UxEvent::StartListen);
    }

    #[test]
    fn gaze_lock_resets_after_grace() {
        let mut lock = GazeLock::with_need(5_000);
        assert_eq!(lock.update(0, true), UxEvent::None);
        assert_eq!(lock.update(100, false), UxEvent::None);
        assert!(lock.since_ms.is_some());
        assert_eq!(lock.update(100 + GAZE_GRACE_MS, false), UxEvent::None);
        assert!(lock.since_ms.is_none());
        assert_eq!(lock.update(200 + GAZE_GRACE_MS, true), UxEvent::None);
        assert_eq!(lock.since_ms, Some(200 + GAZE_GRACE_MS));
    }

    #[test]
    fn hysteresis_stays_on_longer() {
        assert!(gaze_over_hysteresis(false, 100.0, 100.0));
        assert!(!gaze_over_hysteresis(false, 120.0, 100.0));
        assert!(gaze_over_hysteresis(true, 120.0, 100.0));
        assert!(!gaze_over_hysteresis(true, 130.0, 100.0));
    }

    #[test]
    fn chase_gaze_ends_on_buddy() {
        let (x, y) = chase_gaze(10_000, 0, 120.0, 120.0, 800.0, 600.0);
        assert!((x - 120.0).abs() < 1.0 && (y - 120.0).abs() < 1.0);
        let (x0, y0) = chase_gaze(0, 0, 120.0, 120.0, 800.0, 600.0);
        assert!((x0 - 120.0).hypot(y0 - 120.0) > 200.0);
    }

    fn skin_patch(w: u32, h: u32, fx: u32, fy: u32, fw: u32, fh: u32) -> Vec<u8> {
        let mut rgb = vec![20u8; (w * h * 3) as usize];
        for y in fy..fy + fh {
            for x in fx..fx + fw {
                let i = ((y * w + x) * 3) as usize;
                rgb[i] = 190;
                rgb[i + 1] = 120;
                rgb[i + 2] = 100;
            }
        }
        // dark "iris" in the upper-left of the patch
        for y in fy + 2..fy + 8 {
            for x in fx + 6..fx + 12 {
                let i = ((y * w + x) * 3) as usize;
                rgb[i] = 20;
                rgb[i + 1] = 18;
                rgb[i + 2] = 16;
            }
        }
        rgb
    }

    #[test]
    fn skin_blob_finds_face_and_iris() {
        let w = 80u32;
        let h = 60u32;
        let rgb = skin_patch(w, h, 20, 10, 28, 32);
        let g = estimate_face_rgb(&rgb, w, h).expect("face");
        assert!((g.cx - 34.0).abs() < 8.0, "cx {}", g.cx);
        assert!((g.cy - 26.0).abs() < 10.0, "cy {}", g.cy);
        let (ix, iy) = g.look_point();
        assert!(ix < g.cx, "iris should sit in the dark patch, got {ix}");
        assert!(iy < g.cy + 4.0);
    }

    #[test]
    fn empty_frame_has_no_face() {
        let rgb = vec![16u8; 40 * 30 * 3];
        assert!(estimate_face_rgb(&rgb, 40, 30).is_none());
    }
}
