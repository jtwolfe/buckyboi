//! Hand landmarks + gesture classes.
//!
//! Rule classifier works on MediaPipe-style 21 points. Calibration stores
//! per-person centroids (a 1-NN / nearest-centroid model — the “trainable MLP”
//! alternative when you do not want to ship extra ONNX). Optional ONNX palm +
//! landmark models run behind the `hands` feature.

use serde::{Deserialize, Serialize};

pub const GESTURE_ENROLL_NEED: usize = 6;
pub const LANDMARKS: usize = 21;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GestureClass {
    Fist,
    Palm,
    ThumbsUp,
    Point,
    Peace,
    Unknown,
}

impl GestureClass {
    pub fn all() -> [GestureClass; 5] {
        [
            GestureClass::Fist,
            GestureClass::Palm,
            GestureClass::ThumbsUp,
            GestureClass::Point,
            GestureClass::Peace,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            GestureClass::Fist => "FIST",
            GestureClass::Palm => "PALM",
            GestureClass::ThumbsUp => "THUMB",
            GestureClass::Point => "POINT",
            GestureClass::Peace => "PEACE",
            GestureClass::Unknown => "NONE",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "fist" => GestureClass::Fist,
            "palm" | "open" => GestureClass::Palm,
            "thumbs_up" | "thumb" | "thumbs-up" => GestureClass::ThumbsUp,
            "point" | "index" => GestureClass::Point,
            "peace" | "v" => GestureClass::Peace,
            _ => GestureClass::Unknown,
        }
    }

    pub fn next_train(self) -> Option<GestureClass> {
        match self {
            GestureClass::Fist => Some(GestureClass::Palm),
            GestureClass::Palm => Some(GestureClass::ThumbsUp),
            GestureClass::ThumbsUp => Some(GestureClass::Point),
            GestureClass::Point => Some(GestureClass::Peace),
            GestureClass::Peace | GestureClass::Unknown => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GestureAction {
    None,
    Listen,
    Dismiss,
    Confirm,
    Info,
    Settings,
    Mute,
}

impl GestureAction {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "listen" => GestureAction::Listen,
            "dismiss" | "hide" => GestureAction::Dismiss,
            "confirm" => GestureAction::Confirm,
            "info" => GestureAction::Info,
            "settings" => GestureAction::Settings,
            "mute" => GestureAction::Mute,
            _ => GestureAction::None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            GestureAction::None => "none",
            GestureAction::Listen => "listen",
            GestureAction::Dismiss => "dismiss",
            GestureAction::Confirm => "confirm",
            GestureAction::Info => "info",
            GestureAction::Settings => "settings",
            GestureAction::Mute => "mute",
        }
    }
}

pub type GestureMap = [(GestureClass, GestureAction); 5];

pub const DEFAULT_GESTURE_MAP: GestureMap = [
    (GestureClass::Fist, GestureAction::Dismiss),
    (GestureClass::Palm, GestureAction::Listen),
    (GestureClass::ThumbsUp, GestureAction::Confirm),
    (GestureClass::Point, GestureAction::Info),
    (GestureClass::Peace, GestureAction::Settings),
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GestureSample {
    pub class: GestureClass,
    pub landmarks: Vec<f32>,
}

/// 21 points, each (x, y, z) in image-normalized or wrist-relative space.
#[derive(Clone, Debug, PartialEq)]
pub struct HandLandmarks {
    pub pts: [(f32, f32, f32); LANDMARKS],
}

impl HandLandmarks {
    pub fn from_flat(v: &[f32]) -> Option<Self> {
        if v.len() < LANDMARKS * 2 {
            return None;
        }
        let mut pts = [(0.0, 0.0, 0.0); LANDMARKS];
        if v.len() >= LANDMARKS * 3 {
            for i in 0..LANDMARKS {
                pts[i] = (v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
            }
        } else {
            for i in 0..LANDMARKS {
                pts[i] = (v[i * 2], v[i * 2 + 1], 0.0);
            }
        }
        Some(Self { pts })
    }

    pub fn to_flat(&self) -> Vec<f32> {
        let mut o = Vec::with_capacity(LANDMARKS * 3);
        for p in self.pts {
            o.push(p.0);
            o.push(p.1);
            o.push(p.2);
        }
        o
    }

    /// Wrist-relative, scale-normalized (palm width). Good for centroids.
    pub fn normalized(&self) -> Vec<f32> {
        let w = self.pts[0];
        let scale = (self.pts[5].0 - self.pts[17].0)
            .hypot(self.pts[5].1 - self.pts[17].1)
            .max(1e-4);
        let mut o = Vec::with_capacity(LANDMARKS * 2);
        for p in self.pts {
            o.push((p.0 - w.0) / scale);
            o.push((p.1 - w.1) / scale);
        }
        o
    }
}

fn dist(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

fn finger_extended(hand: &HandLandmarks, tip: usize, pip: usize, mcp: usize) -> bool {
    dist(hand.pts[tip], hand.pts[mcp]) > dist(hand.pts[pip], hand.pts[mcp]) * 1.15
}

/// Classic MediaPipe-style rules. Indices: 0 wrist, 4 thumb tip, 8 index, …
pub fn classify_rules(hand: &HandLandmarks) -> (GestureClass, f32) {
    let thumb = finger_extended(hand, 4, 3, 2);
    let index = finger_extended(hand, 8, 6, 5);
    let middle = finger_extended(hand, 12, 10, 9);
    let ring = finger_extended(hand, 16, 14, 13);
    let pinky = finger_extended(hand, 20, 18, 17);
    let ext = [thumb, index, middle, ring, pinky]
        .iter()
        .filter(|b| **b)
        .count();
    if !index && !middle && !ring && !pinky {
        if thumb && hand.pts[4].1 < hand.pts[2].1 - 0.08 {
            return (GestureClass::ThumbsUp, 0.82);
        }
        return (GestureClass::Fist, 0.86);
    }
    if index && middle && !ring && !pinky {
        return (GestureClass::Peace, 0.84);
    }
    if index && !middle && !ring && !pinky {
        return (GestureClass::Point, 0.88);
    }
    if ext >= 4 {
        return (GestureClass::Palm, 0.85);
    }
    (GestureClass::Unknown, 0.2)
}

pub fn gesture_centroid(samples: &[Vec<f32>]) -> Option<Vec<f32>> {
    if samples.is_empty() {
        return None;
    }
    let dim = samples[0].len();
    if dim == 0 || samples.iter().any(|s| s.len() != dim) {
        return None;
    }
    let n = samples.len() as f32;
    let mut acc = vec![0.0f32; dim];
    for s in samples {
        for i in 0..dim {
            acc[i] += s[i];
        }
    }
    for v in &mut acc {
        *v /= n;
    }
    Some(acc)
}

fn cosine_vec(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = (na.sqrt() * nb.sqrt()).max(1e-9);
    Some((dot / d).clamp(-1.0, 1.0))
}

/// Nearest-centroid classifier from calibration samples.
pub fn classify_trained(
    hand: &HandLandmarks,
    centroids: &[(GestureClass, Vec<f32>)],
    threshold: f32,
) -> (GestureClass, f32) {
    let feat = hand.normalized();
    let mut best = GestureClass::Unknown;
    let mut best_s = -1.0f32;
    for (cls, c) in centroids {
        if let Some(s) = cosine_vec(&feat, c) {
            if s > best_s {
                best_s = s;
                best = *cls;
            }
        }
    }
    if best_s < threshold {
        (GestureClass::Unknown, best_s.max(0.0))
    } else {
        (best, best_s)
    }
}

pub fn classify_gesture(
    hand: &HandLandmarks,
    centroids: &[(GestureClass, Vec<f32>)],
    threshold: f32,
) -> (GestureClass, f32) {
    if !centroids.is_empty() {
        let (c, s) = classify_trained(hand, centroids, threshold);
        if c != GestureClass::Unknown {
            return (c, s);
        }
    }
    classify_rules(hand)
}

pub fn action_for(class: GestureClass, map: &GestureMap) -> GestureAction {
    map.iter()
        .find(|(c, _)| *c == class)
        .map(|(_, a)| *a)
        .unwrap_or(GestureAction::None)
}

/// Synthetic hand for tests / VM demos (`BUCKYBOI_HAND_SIM`).
pub fn synthetic(class: GestureClass) -> HandLandmarks {
    let mut pts = [(0.0, 0.0, 0.0); LANDMARKS];
    pts[0] = (0.5, 0.7, 0.0);
    // rough palm
    pts[5] = (0.42, 0.55, 0.0);
    pts[9] = (0.50, 0.54, 0.0);
    pts[13] = (0.58, 0.55, 0.0);
    pts[17] = (0.64, 0.58, 0.0);
    let extend = |pts: &mut [(f32, f32, f32); 21], mcp: usize, pip: usize, tip: usize, up: bool| {
        let base = pts[mcp];
        if up {
            pts[pip] = (base.0, base.1 - 0.08, 0.0);
            pts[tip] = (base.0, base.1 - 0.18, 0.0);
        } else {
            // Tip closer to MCP than PIP so finger_extended is false.
            pts[pip] = (base.0 + 0.02, base.1 + 0.04, 0.0);
            pts[tip] = (base.0 + 0.01, base.1 + 0.015, 0.0);
        }
    };
    let (t, i, m, r, p) = match class {
        GestureClass::Palm => (true, true, true, true, true),
        GestureClass::Fist => (false, false, false, false, false),
        GestureClass::ThumbsUp => (true, false, false, false, false),
        GestureClass::Point => (false, true, false, false, false),
        GestureClass::Peace => (false, true, true, false, false),
        GestureClass::Unknown => (false, false, false, true, false),
    };
    // thumb
    pts[2] = (0.38, 0.60, 0.0);
    if t {
        pts[3] = (0.36, 0.48, 0.0);
        pts[4] = (0.35, 0.38, 0.0);
    } else {
        pts[3] = (0.40, 0.63, 0.0);
        pts[4] = (0.385, 0.61, 0.0);
    }
    extend(&mut pts, 5, 6, 8, i);
    extend(&mut pts, 9, 10, 12, m);
    extend(&mut pts, 13, 14, 16, r);
    extend(&mut pts, 17, 18, 20, p);
    HandLandmarks { pts }
}

pub fn palm_model_names() -> &'static [&'static str] {
    &[
        "palm_detection.onnx",
        "palm_detection_mediapipe_2023feb.onnx",
        "palm_detection_full.onnx",
        "palm.onnx",
    ]
}

pub fn landmark_model_names() -> &'static [&'static str] {
    &[
        "hand_landmark.onnx",
        "handpose_estimation_mediapipe_2023feb.onnx",
        "hand_landmark_full.onnx",
        "hand_landmark_lite.onnx",
    ]
}

/// Two-stage palm → landmark when `--features hands` and models exist.
#[cfg(feature = "hands")]
pub fn extract_landmarks_onnx(rgb: &[u8], w: u32, h: u32) -> Option<HandLandmarks> {
    onnx::detect(rgb, w, h)
}

#[cfg(not(feature = "hands"))]
pub fn extract_landmarks_onnx(_rgb: &[u8], _w: u32, _h: u32) -> Option<HandLandmarks> {
    None
}

#[cfg(feature = "hands")]
mod onnx {
    use super::*;
    use crate::identity::models_dir;
    use crate::identity::palm::{
        decode_palms, nms_palms, palm_anchors_192, palm_to_roi, roi_to_frame, warp_roi_rgb, Anchor,
        HAND_LANDMARK_SIZE, PALM_INPUT,
    };
    use ndarray::Array4;
    use std::path::PathBuf;
    use std::sync::Mutex;

    const PALM_IN: usize = PALM_INPUT as usize;
    const LM_IN: usize = HAND_LANDMARK_SIZE as usize;

    struct Nets {
        palm: ort::session::Session,
        landmark: Option<ort::session::Session>,
        anchors: Vec<Anchor>,
    }

    static NETS: Mutex<Option<Nets>> = Mutex::new(None);

    fn find(names: &[&str]) -> Option<PathBuf> {
        let dir = models_dir()?;
        names.iter().map(|n| dir.join(n)).find(|p| p.is_file())
    }

    fn session() -> Option<()> {
        let mut g = NETS.lock().ok()?;
        if g.is_some() {
            return Some(());
        }
        let palm_path = find(palm_model_names())?;
        let palm = ort::session::Session::builder()
            .ok()?
            .commit_from_file(&palm_path)
            .ok()?;
        let landmark = find(landmark_model_names()).and_then(|p| {
            ort::session::Session::builder()
                .ok()?
                .commit_from_file(&p)
                .ok()
        });
        eprintln!(
            "buckyboi: hands ONNX palm={} landmark={}",
            palm_path.display(),
            if landmark.is_some() { "yes" } else { "no" }
        );
        *g = Some(Nets {
            palm,
            landmark,
            anchors: palm_anchors_192(),
        });
        Some(())
    }

    fn rgb_blob(rgb: &[u8], w: u32, h: u32, size: usize, scale_127: bool) -> Array4<f32> {
        let mut blob = Array4::<f32>::zeros((1, 3, size, size));
        if w == 0 || h == 0 {
            return blob;
        }
        for y in 0..size {
            let sy = (y as f32 * h as f32 / size as f32).clamp(0.0, h as f32 - 1.0) as u32;
            for x in 0..size {
                let sx = (x as f32 * w as f32 / size as f32).clamp(0.0, w as f32 - 1.0) as u32;
                let i = ((sy * w + sx) * 3) as usize;
                if i + 2 >= rgb.len() {
                    continue;
                }
                let (r, g, b) = (rgb[i] as f32, rgb[i + 1] as f32, rgb[i + 2] as f32);
                let (nr, ng, nb) = if scale_127 {
                    (
                        (r - 127.5) / 127.5,
                        (g - 127.5) / 127.5,
                        (b - 127.5) / 127.5,
                    )
                } else {
                    (r / 255.0, g / 255.0, b / 255.0)
                };
                blob[[0, 0, y, x]] = nr;
                blob[[0, 1, y, x]] = ng;
                blob[[0, 2, y, x]] = nb;
            }
        }
        blob
    }

    fn collect_f32(outputs: &ort::session::SessionOutputs<'_>) -> Vec<Vec<f32>> {
        let mut v = Vec::new();
        for (_, t) in outputs.iter() {
            if let Ok((_shape, data)) = t.try_extract_tensor::<f32>() {
                v.push(data.to_vec());
            }
        }
        v
    }

    fn landmarks_from_flat(
        data: &[f32],
        roi: Option<crate::identity::palm::HandRoi>,
    ) -> Option<HandLandmarks> {
        if data.len() < 42 {
            return None;
        }
        let trip = data.len() >= 63;
        let mut pts = [(0.0, 0.0, 0.0); LANDMARKS];
        for i in 0..LANDMARKS {
            let (mut x, mut y, z) = if trip {
                (data[i * 3], data[i * 3 + 1], data[i * 3 + 2])
            } else {
                (data[i * 2], data[i * 2 + 1], 0.0)
            };
            // Crop-pixel vs unit interval.
            if x <= 1.5 && y <= 1.5 {
                x *= LM_IN as f32;
                y *= LM_IN as f32;
            }
            if let Some(roi) = roi {
                let (fx, fy) = roi_to_frame(roi, LM_IN as f32, x, y);
                pts[i] = (fx, fy, z);
            } else {
                pts[i] = (x, y, z);
            }
        }
        Some(HandLandmarks { pts })
    }

    pub fn detect(rgb: &[u8], w: u32, h: u32) -> Option<HandLandmarks> {
        session()?;
        let mut g = NETS.lock().ok()?;
        let nets = g.as_mut()?;

        let blob = rgb_blob(rgb, w, h, PALM_IN, true);
        let input = ort::value::Tensor::from_array(blob).ok()?;
        let outputs = nets.palm.run(ort::inputs![input]).ok()?;
        let tensors = collect_f32(&outputs);

        // Landmark-only file dropped in as "palm": 42/63 floats.
        for t in &tensors {
            if (t.len() == 63 || t.len() == 42) && nets.landmark.is_none() {
                return landmarks_from_flat(t, None);
            }
        }

        let (regs, scores) = match tensors.len() {
            0 => return None,
            1 => {
                // Some exports concat [N, 19] = 18 + score.
                let t = &tensors[0];
                if t.len() % 19 == 0 {
                    let n = t.len() / 19;
                    let mut r = Vec::with_capacity(n * 18);
                    let mut s = Vec::with_capacity(n);
                    for i in 0..n {
                        r.extend_from_slice(&t[i * 19..i * 19 + 18]);
                        s.push(t[i * 19 + 18]);
                    }
                    (r, s)
                } else {
                    return None;
                }
            }
            _ => {
                let a = &tensors[0];
                let b = &tensors[1];
                if a.len() >= b.len() * 8 {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                }
            }
        };

        let mut dets = decode_palms(&regs, &scores, &nets.anchors, PALM_IN as f32, 0.55);
        // Scale letterboxed 192 coords back to the frame (stretch).
        for d in &mut dets {
            let sx = w as f32 / PALM_IN as f32;
            let sy = h as f32 / PALM_IN as f32;
            d.x *= sx;
            d.y *= sy;
            d.w *= sx;
            d.h *= sy;
            for k in &mut d.kps {
                k.0 *= sx;
                k.1 *= sy;
            }
        }
        let dets = nms_palms(dets, 0.3);
        let palm = dets.first()?;
        let roi = palm_to_roi(palm, w as f32, h as f32);

        if let Some(lm) = nets.landmark.as_mut() {
            let crop = warp_roi_rgb(rgb, w, h, roi, LM_IN);
            let blob = rgb_blob(&crop, LM_IN as u32, LM_IN as u32, LM_IN, true);
            let input = ort::value::Tensor::from_array(blob).ok()?;
            let outputs = lm.run(ort::inputs![input]).ok()?;
            for t in collect_f32(&outputs) {
                if t.len() >= 42 {
                    return landmarks_from_flat(&t, Some(roi));
                }
            }
        }

        // Palm keypoints only: map 7 pts onto a coarse 21-pt skeleton so
        // the rule classifier still has a wrist + MCP + tips to work with.
        let mut pts = [(0.0, 0.0, 0.0); LANDMARKS];
        pts[0] = (palm.kps[0].0, palm.kps[0].1, 0.0);
        for (i, dst) in [5usize, 9, 13, 17].iter().enumerate() {
            if i + 1 < palm.kps.len() {
                pts[*dst] = (palm.kps[i + 1].0, palm.kps[i + 1].1, 0.0);
            }
        }
        Some(HandLandmarks { pts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_recognize_synthetic() {
        for cls in GestureClass::all() {
            let (got, score) = classify_rules(&synthetic(cls));
            assert_eq!(got, cls, "{cls:?} score={score}");
            assert!(score > 0.7);
        }
    }

    #[test]
    fn centroid_trains_and_matches() {
        let mut samples = Vec::new();
        for cls in GestureClass::all() {
            let mut xs = Vec::new();
            for _ in 0..4 {
                xs.push(synthetic(cls).normalized());
            }
            samples.push((cls, gesture_centroid(&xs).unwrap()));
        }
        let (got, s) = classify_trained(&synthetic(GestureClass::Point), &samples, 0.7);
        assert_eq!(got, GestureClass::Point);
        assert!(s > 0.85);
    }

    #[test]
    fn unknown_below_threshold() {
        let fist = synthetic(GestureClass::Fist).normalized();
        let cents = vec![(
            GestureClass::Palm,
            synthetic(GestureClass::Palm).normalized(),
        )];
        let hand = HandLandmarks::from_flat(&fist).unwrap_or(synthetic(GestureClass::Fist));
        // Use a real fist hand against a palm-only gallery.
        let fist_hand = synthetic(GestureClass::Fist);
        let (got, _) = classify_trained(&fist_hand, &cents, 0.99);
        let _ = (hand, fist);
        assert_eq!(got, GestureClass::Unknown);
    }

    #[test]
    fn default_map_fist_dismisses() {
        assert_eq!(
            action_for(GestureClass::Fist, &DEFAULT_GESTURE_MAP),
            GestureAction::Dismiss
        );
    }
}
