//! 5-point affine / 9-point ridge gaze calibration (pure).
//!
//! Persists `~/.config/buckyboi/gaze_calib.json`. Screen-size mismatch
//! marks the file stale; a high RMSE refuses to overwrite it.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::path::PathBuf;

pub const CALIB_VERSION: u32 = 1;
pub const RIDGE_LAMBDA: f32 = 1e-2;
pub const RMSE_PX_HARD: f32 = 200.0;
pub const RMSE_FRAC: f32 = 0.15;
pub const ACQUIRE_MS: u64 = 1_000;
pub const SAMPLE_MS: u64 = 800;
pub const NEED_SNAPS: usize = 8;
pub const DOT_INSET: f32 = 0.10;

const FIVE_HINTS: [&str; 5] = ["LOOK CENTER", "LOOK TL", "LOOK TR", "LOOK BR", "LOOK BL"];

/// One observation used to fit affine / ridge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibObs {
    pub iris_nx: f32,
    pub iris_ny: f32,
    pub nose_nx: f32,
    pub nose_ny: f32,
    pub sx: f32,
    pub sy: f32,
}

/// Raw mesh features sampled during a dwell.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CalibFeat {
    pub iris_nx: f32,
    pub iris_ny: f32,
    pub nose_nx: f32,
    pub nose_ny: f32,
    pub both_irises: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GazeCalib {
    pub version: u32,
    pub screen_w: u32,
    pub screen_h: u32,
    pub camera: String,
    pub model: String,
    pub points: u32,
    #[serde(default)]
    pub affine: Option<[f32; 6]>,
    #[serde(default)]
    pub ridge_w: Option<[f32; 10]>,
    pub rmse_px: f32,
    pub created_ms: u64,
}

impl GazeCalib {
    pub fn is_stale(&self, sw: u32, sh: u32) -> bool {
        self.screen_w != sw || self.screen_h != sh
    }

    pub fn apply(
        &self,
        iris_nx: f32,
        iris_ny: f32,
        nose_nx: f32,
        nose_ny: f32,
    ) -> Option<(f32, f32)> {
        if self.points == 9 {
            let w = self.ridge_w?;
            let sx = w[0] * iris_nx + w[1] * iris_ny + w[2] * nose_nx + w[3] * nose_ny + w[4];
            let sy = w[5] * iris_nx + w[6] * iris_ny + w[7] * nose_nx + w[8] * nose_ny + w[9];
            return Some((sx, sy));
        }
        let a = self.affine?;
        let sx = a[0] * iris_nx + a[1] * iris_ny + a[2];
        let sy = a[3] * iris_nx + a[4] * iris_ny + a[5];
        Some((sx, sy))
    }

    pub fn save(&self) -> bool {
        let Some(path) = calib_path() else {
            return false;
        };
        save_to(self, &path)
    }

    pub fn load() -> Option<Self> {
        load_from(&calib_path()?)
    }
}

pub fn calib_path() -> Option<PathBuf> {
    Some(crate::identity::config_dir()?.join("gaze_calib.json"))
}

pub fn camera_label() -> String {
    crate::identity::env_or_alias("BUCKYBOI_CAMERA", "BUDDY_CAMERA")
        .unwrap_or_else(|| "/dev/video0".into())
}

pub fn calib_point_count() -> usize {
    match std::env::var("BUCKYBOI_GAZE_CALIB_POINTS").ok().as_deref() {
        Some(v) if v.trim() == "9" => 9,
        _ => 5,
    }
}

pub fn rmse_cap(sw: f32, sh: f32) -> f32 {
    RMSE_PX_HARD.min(RMSE_FRAC * sw.min(sh))
}

pub fn rmse_ok(rmse: f32, sw: f32, sh: f32) -> bool {
    rmse.is_finite() && rmse <= rmse_cap(sw, sh)
}

/// Target screen positions, inset 10%. Order for 5-pt: center, TL, TR, BR, BL.
pub fn dot_pos(i: usize, n: usize, sw: f32, sh: f32) -> (f32, f32) {
    let mx = sw * DOT_INSET;
    let my = sh * DOT_INSET;
    if n == 9 {
        let r = (i / 3) as f32;
        let c = (i % 3) as f32;
        return (
            mx + c * (sw - 2.0 * mx) / 2.0,
            my + r * (sh - 2.0 * my) / 2.0,
        );
    }
    match i {
        0 => (sw * 0.5, sh * 0.5),
        1 => (mx, my),
        2 => (sw - mx, my),
        3 => (sw - mx, sh - my),
        4 => (mx, sh - my),
        _ => (sw * 0.5, sh * 0.5),
    }
}

pub fn look_hint(i: usize, n: usize) -> &'static str {
    if n == 9 {
        const NINE: [&str; 9] = [
            "LOOK 1", "LOOK 2", "LOOK 3", "LOOK 4", "LOOK 5", "LOOK 6", "LOOK 7", "LOOK 8",
            "LOOK 9",
        ];
        return NINE.get(i).copied().unwrap_or("LOOK");
    }
    FIVE_HINTS.get(i).copied().unwrap_or("LOOK")
}

pub fn load_from(path: &Path) -> Option<GazeCalib> {
    let txt = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&txt).ok()
}

pub fn save_to(calib: &GazeCalib, path: &Path) -> bool {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let Ok(txt) = serde_json::to_string_pretty(calib) else {
        return false;
    };
    std::fs::write(path, txt).is_ok()
}

/// Least-squares 2×3 affine on `(iris_nx, iris_ny)` → `(sx, sy)`.
pub fn fit_affine(samples: &[CalibObs]) -> Option<([f32; 6], f32)> {
    if samples.len() < 3 {
        return None;
    }
    let n = samples.len();
    let mut xtx = [[0.0f64; 3]; 3];
    let mut xty_x = [0.0f64; 3];
    let mut xty_y = [0.0f64; 3];
    for s in samples {
        let row = [s.iris_nx as f64, s.iris_ny as f64, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                xtx[i][j] += row[i] * row[j];
            }
            xty_x[i] += row[i] * s.sx as f64;
            xty_y[i] += row[i] * s.sy as f64;
        }
    }
    let a = solve_n(&xtx, &xty_x, 3)?;
    let d = solve_n(&xtx, &xty_y, 3)?;
    let affine = [
        a[0] as f32,
        a[1] as f32,
        a[2] as f32,
        d[0] as f32,
        d[1] as f32,
        d[2] as f32,
    ];
    let mut se = 0.0f64;
    for s in samples {
        let px = affine[0] as f64 * s.iris_nx as f64
            + affine[1] as f64 * s.iris_ny as f64
            + affine[2] as f64;
        let py = affine[3] as f64 * s.iris_nx as f64
            + affine[4] as f64 * s.iris_ny as f64
            + affine[5] as f64;
        se += (px - s.sx as f64).powi(2) + (py - s.sy as f64).powi(2);
    }
    let rmse = (se / n as f64).sqrt() as f32;
    Some((affine, rmse))
}

/// Ridge 2×5 on `[iris_nx, iris_ny, nose_nx, nose_ny, 1]`, `λ = 1e-2`.
pub fn fit_ridge(samples: &[CalibObs]) -> Option<([f32; 10], f32)> {
    if samples.len() < 5 {
        return None;
    }
    let n = samples.len();
    let lam = RIDGE_LAMBDA as f64;
    let mut xtx = [[0.0f64; 5]; 5];
    let mut xty_x = [0.0f64; 5];
    let mut xty_y = [0.0f64; 5];
    for s in samples {
        let row = [
            s.iris_nx as f64,
            s.iris_ny as f64,
            s.nose_nx as f64,
            s.nose_ny as f64,
            1.0,
        ];
        for i in 0..5 {
            for j in 0..5 {
                xtx[i][j] += row[i] * row[j];
            }
            xty_x[i] += row[i] * s.sx as f64;
            xty_y[i] += row[i] * s.sy as f64;
        }
    }
    for i in 0..5 {
        xtx[i][i] += lam;
    }
    let wx = solve_n(&xtx, &xty_x, 5)?;
    let wy = solve_n(&xtx, &xty_y, 5)?;
    let mut w = [0.0f32; 10];
    for i in 0..5 {
        w[i] = wx[i] as f32;
        w[5 + i] = wy[i] as f32;
    }
    let mut se = 0.0f64;
    for s in samples {
        let px = w[0] as f64 * s.iris_nx as f64
            + w[1] as f64 * s.iris_ny as f64
            + w[2] as f64 * s.nose_nx as f64
            + w[3] as f64 * s.nose_ny as f64
            + w[4] as f64;
        let py = w[5] as f64 * s.iris_nx as f64
            + w[6] as f64 * s.iris_ny as f64
            + w[7] as f64 * s.nose_nx as f64
            + w[8] as f64 * s.nose_ny as f64
            + w[9] as f64;
        se += (px - s.sx as f64).powi(2) + (py - s.sy as f64).powi(2);
    }
    let rmse = (se / n as f64).sqrt() as f32;
    Some((w, rmse))
}

pub fn from_affine(samples: &[CalibObs], sw: u32, sh: u32, camera: &str) -> Option<GazeCalib> {
    let (affine, rmse) = fit_affine(samples)?;
    if !rmse_ok(rmse, sw as f32, sh as f32) {
        return None;
    }
    Some(GazeCalib {
        version: CALIB_VERSION,
        screen_w: sw,
        screen_h: sh,
        camera: camera.to_string(),
        model: "face_landmarker".into(),
        points: 5,
        affine: Some(affine),
        ridge_w: None,
        rmse_px: rmse,
        created_ms: now_ms(),
    })
}

pub fn from_ridge(samples: &[CalibObs], sw: u32, sh: u32, camera: &str) -> Option<GazeCalib> {
    let (w, rmse) = fit_ridge(samples)?;
    if !rmse_ok(rmse, sw as f32, sh as f32) {
        return None;
    }
    Some(GazeCalib {
        version: CALIB_VERSION,
        screen_w: sw,
        screen_h: sh,
        camera: camera.to_string(),
        model: "face_landmarker".into(),
        points: 9,
        affine: None,
        ridge_w: Some(w),
        rmse_px: rmse,
        created_ms: now_ms(),
    })
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn solve_n<const N: usize>(a_in: &[[f64; N]; N], b_in: &[f64; N], n: usize) -> Option<[f64; N]> {
    let mut a = [[0.0f64; N]; N];
    let mut b = [0.0f64; N];
    for i in 0..n {
        a[i] = a_in[i];
        b[i] = b_in[i];
    }
    for k in 0..n {
        let mut piv = k;
        let mut best = a[k][k].abs();
        for i in k + 1..n {
            let v = a[i][k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }
        let diag = a[k][k];
        for j in k..n {
            a[k][j] /= diag;
        }
        b[k] /= diag;
        for i in 0..n {
            if i == k {
                continue;
            }
            let f = a[i][k];
            for j in k..n {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut out = [0.0f64; N];
    out[..n].copy_from_slice(&b[..n]);
    Some(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CalibPhase {
    Acquire,
    Sample,
}

/// Overlay-owned 5/9-dot dwell sequence. Counts as a panel (`calib_open`).
#[derive(Clone, Debug)]
pub struct GazeCalibSession {
    pub open: bool,
    n_dots: usize,
    idx: usize,
    retried: bool,
    phase: CalibPhase,
    started_ms: u64,
    snaps: Vec<[f32; 4]>,
    collected: Vec<CalibObs>,
    hint: String,
    progress: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CalibEvent {
    None,
    Finished(Vec<CalibObs>),
    Failed,
}

impl Default for GazeCalibSession {
    fn default() -> Self {
        Self::new()
    }
}

impl GazeCalibSession {
    pub fn new() -> Self {
        Self {
            open: false,
            n_dots: 5,
            idx: 0,
            retried: false,
            phase: CalibPhase::Acquire,
            started_ms: 0,
            snaps: Vec::new(),
            collected: Vec::new(),
            hint: String::new(),
            progress: 0.0,
        }
    }

    pub fn begin(&mut self, now_ms: u64) {
        self.open = true;
        self.n_dots = calib_point_count().clamp(5, 9);
        if self.n_dots != 9 {
            self.n_dots = 5;
        }
        self.idx = 0;
        self.retried = false;
        self.phase = CalibPhase::Acquire;
        self.started_ms = now_ms;
        self.snaps.clear();
        self.collected.clear();
        self.hint = look_hint(0, self.n_dots).into();
        self.progress = 0.0;
    }

    pub fn cancel(&mut self) {
        *self = Self::new();
    }

    pub fn hint(&self) -> &str {
        &self.hint
    }

    pub fn progress(&self) -> f32 {
        self.progress
    }

    pub fn n_dots(&self) -> usize {
        self.n_dots
    }

    pub fn current_idx(&self) -> usize {
        self.idx
    }

    pub fn min_points(&self) -> usize {
        if self.n_dots == 9 {
            6
        } else {
            4
        }
    }

    pub fn skip_current(&mut self, now_ms: u64) -> CalibEvent {
        if !self.open {
            return CalibEvent::None;
        }
        self.advance(now_ms)
    }

    pub fn tick(&mut self, now_ms: u64, feat: Option<CalibFeat>, sw: f32, sh: f32) -> CalibEvent {
        if !self.open {
            return CalibEvent::None;
        }
        let elapsed = now_ms.saturating_sub(self.started_ms);
        match self.phase {
            CalibPhase::Acquire => {
                self.hint = look_hint(self.idx, self.n_dots).into();
                self.progress = (elapsed as f32 / ACQUIRE_MS as f32).clamp(0.0, 1.0);
                if elapsed >= ACQUIRE_MS {
                    self.phase = CalibPhase::Sample;
                    self.started_ms = now_ms;
                    self.snaps.clear();
                    self.hint = "HOLD".into();
                    self.progress = 0.0;
                }
            }
            CalibPhase::Sample => {
                self.hint = "HOLD".into();
                self.progress = (elapsed as f32 / SAMPLE_MS as f32).clamp(0.0, 1.0);
                if let Some(f) = feat {
                    if f.both_irises {
                        self.snaps
                            .push([f.iris_nx, f.iris_ny, f.nose_nx, f.nose_ny]);
                    }
                }
                if elapsed >= SAMPLE_MS {
                    if self.snaps.len() >= NEED_SNAPS {
                        let (sx, sy) = dot_pos(self.idx, self.n_dots, sw, sh);
                        self.collected.push(mean_obs(&self.snaps, sx, sy));
                        return self.advance(now_ms);
                    }
                    if !self.retried {
                        self.retried = true;
                        self.phase = CalibPhase::Acquire;
                        self.started_ms = now_ms;
                        self.snaps.clear();
                        self.hint = look_hint(self.idx, self.n_dots).into();
                        self.progress = 0.0;
                    } else {
                        return self.advance(now_ms);
                    }
                }
            }
        }
        CalibEvent::None
    }

    fn advance(&mut self, now_ms: u64) -> CalibEvent {
        self.idx += 1;
        self.retried = false;
        self.snaps.clear();
        if self.idx >= self.n_dots {
            let enough = self.collected.len() >= self.min_points();
            self.open = false;
            self.hint = if enough {
                "GAZE OK".into()
            } else {
                "RECALIBRATE".into()
            };
            self.progress = 1.0;
            return if enough {
                CalibEvent::Finished(std::mem::take(&mut self.collected))
            } else {
                CalibEvent::Failed
            };
        }
        self.phase = CalibPhase::Acquire;
        self.started_ms = now_ms;
        self.hint = look_hint(self.idx, self.n_dots).into();
        self.progress = 0.0;
        CalibEvent::None
    }
}

fn mean_obs(snaps: &[[f32; 4]], sx: f32, sy: f32) -> CalibObs {
    let n = snaps.len().max(1) as f32;
    let mut a = [0.0f32; 4];
    for s in snaps {
        for i in 0..4 {
            a[i] += s[i];
        }
    }
    CalibObs {
        iris_nx: a[0] / n,
        iris_ny: a[1] / n,
        nose_nx: a[2] / n,
        nose_ny: a[3] / n,
        sx,
        sy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn five_identity(sw: f32, sh: f32) -> Vec<CalibObs> {
        (0..5)
            .map(|i| {
                let (sx, sy) = dot_pos(i, 5, sw, sh);
                CalibObs {
                    iris_nx: sx / sw,
                    iris_ny: sy / sh,
                    nose_nx: 0.5,
                    nose_ny: 0.5,
                    sx,
                    sy,
                }
            })
            .collect()
    }

    #[test]
    fn affine_recovers_identity() {
        let sw = 1920.0;
        let sh = 1080.0;
        let samples = five_identity(sw, sh);
        let (a, rmse) = fit_affine(&samples).expect("fit");
        assert!(rmse < 1.0, "rmse={rmse}");
        assert!((a[0] - sw).abs() < 1e-2, "a={:?}", a);
        assert!(a[1].abs() < 1e-2);
        assert!(a[2].abs() < 1e-1);
        assert!(a[3].abs() < 1e-2);
        assert!((a[4] - sh).abs() < 1e-2);
        assert!(a[5].abs() < 1e-1);
        let c = from_affine(&samples, sw as u32, sh as u32, "/dev/video0").unwrap();
        let (sx, sy) = c.apply(0.5, 0.5, 0.5, 0.5).unwrap();
        assert!((sx - 960.0).abs() < 1.0 && (sy - 540.0).abs() < 1.0);
    }

    #[test]
    fn ridge_layout_is_2x5() {
        let sw = 1920.0f32;
        let sh = 1080.0;
        let mut samples = Vec::new();
        for i in 0..9 {
            let (sx0, sy0) = dot_pos(i, 9, sw, sh);
            let inx = 0.2 + (i % 3) as f32 * 0.25;
            let iny = 0.2 + (i / 3) as f32 * 0.25;
            let nnx = 0.4 + (i % 3) as f32 * 0.1;
            let nny = 0.45 + (i / 3) as f32 * 0.05;
            // Known 2×5 map (bias last).
            let sx = 800.0 * inx + 10.0 * iny + 200.0 * nnx + 5.0 * nny + 40.0;
            let sy = 20.0 * inx + 700.0 * iny + 8.0 * nnx + 120.0 * nny + 30.0;
            let _ = (sx0, sy0);
            samples.push(CalibObs {
                iris_nx: inx,
                iris_ny: iny,
                nose_nx: nnx,
                nose_ny: nny,
                sx,
                sy,
            });
        }
        let (w, rmse) = fit_ridge(&samples).expect("ridge");
        assert_eq!(w.len(), 10);
        assert!(rmse < 8.0, "rmse={rmse}");
        let c = from_ridge(&samples, sw as u32, sh as u32, "/dev/video0").unwrap();
        assert_eq!(c.points, 9);
        assert!(c.affine.is_none());
        assert!(c.ridge_w.is_some());
        let s = &samples[0];
        let (px, py) = c.apply(s.iris_nx, s.iris_ny, s.nose_nx, s.nose_ny).unwrap();
        assert!((px - s.sx).abs() < 8.0 && (py - s.sy).abs() < 8.0);
    }

    #[test]
    fn rmse_rejects_over_cap() {
        let sw = 1920.0;
        let sh = 1054.0;
        let cap = rmse_cap(sw, sh);
        assert!((cap - 158.1).abs() < 0.2, "cap={cap}");
        assert!(!rmse_ok(200.0, sw, sh));
        assert!(rmse_ok(40.0, sw, sh));
        let mut samples = five_identity(sw, sh);
        for s in &mut samples {
            s.sx += 400.0;
            s.sy += 400.0;
        }
        // Affine can absorb a uniform translation via c,f — scramble instead.
        for (i, s) in samples.iter_mut().enumerate() {
            s.sx = if i % 2 == 0 { 0.0 } else { sw };
            s.sy = if i % 3 == 0 { 0.0 } else { sh };
        }
        assert!(from_affine(&samples, sw as u32, sh as u32, "/dev/video0").is_none());
    }

    #[test]
    fn stale_on_screen_mismatch() {
        let c = GazeCalib {
            version: 1,
            screen_w: 1920,
            screen_h: 1054,
            camera: "/dev/video0".into(),
            model: "face_landmarker".into(),
            points: 5,
            affine: Some([1920.0, 0.0, 0.0, 0.0, 1054.0, 0.0]),
            ridge_w: None,
            rmse_px: 40.0,
            created_ms: 0,
        };
        assert!(!c.is_stale(1920, 1054));
        assert!(c.is_stale(1280, 720));
        assert!(c.is_stale(1920, 1080));
    }

    #[test]
    fn json_roundtrip() {
        let dir = std::env::temp_dir().join(format!("buckyboi-calib-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("gaze_calib.json");
        let c = GazeCalib {
            version: 1,
            screen_w: 1920,
            screen_h: 1054,
            camera: "/dev/video0".into(),
            model: "face_landmarker".into(),
            points: 5,
            affine: Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ridge_w: None,
            rmse_px: 48.2,
            created_ms: 7,
        };
        assert!(save_to(&c, &path));
        let got = load_from(&path).expect("load");
        assert_eq!(got.affine, c.affine);
        assert_eq!(got.rmse_px, 48.2);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn session_needs_both_irises() {
        let mut s = GazeCalibSession::new();
        s.begin(0);
        assert_eq!(s.hint(), "LOOK CENTER");
        // Acquire 1s, then sample without irises → retry, then skip.
        assert!(matches!(
            s.tick(500, None, 1920.0, 1080.0),
            CalibEvent::None
        ));
        let _ = s.tick(1_000, None, 1920.0, 1080.0);
        assert_eq!(s.hint(), "HOLD");
        let feat = CalibFeat {
            both_irises: false,
            ..CalibFeat::default()
        };
        let _ = s.tick(1_800, Some(feat), 1920.0, 1080.0);
        // retried acquire
        assert_eq!(s.current_idx(), 0);
    }

    #[test]
    fn five_dots_inset_order() {
        let (cx, cy) = dot_pos(0, 5, 1000.0, 1000.0);
        assert!((cx - 500.0).abs() < 1e-3 && (cy - 500.0).abs() < 1e-3);
        let (tlx, tly) = dot_pos(1, 5, 1000.0, 1000.0);
        assert!((tlx - 100.0).abs() < 1e-3 && (tly - 100.0).abs() < 1e-3);
        let (trx, _) = dot_pos(2, 5, 1000.0, 1000.0);
        assert!((trx - 900.0).abs() < 1e-3);
        assert_eq!(look_hint(0, 5), "LOOK CENTER");
        assert!(look_hint(1, 5).len() <= 16);
    }
}
