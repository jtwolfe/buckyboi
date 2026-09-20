//! 478-pt Face Landmarker iris-in-eye features (`feature = "face"`).
//!
//! SCRFD box → square ROI (margin 0.25, optional eye-level rotate) →
//! `face_landmarker.onnx` NCHW RGB/255 256². Face-box fallback lives in
//! the worker when this returns `ok = false`.

use crate::identity::scrfd::DetectedFace;

pub const LANDMARKER_SIZE: u32 = 256;
pub const ROI_MARGIN: f32 = 0.25;
pub const IRIS_L: usize = 468;
pub const IRIS_R: usize = 473;
pub const NOSE_TIP: usize = 1;
/// Outer, inner, upper lid, lower lid.
pub const LEFT_EYE: [usize; 4] = [33, 133, 159, 145];
/// Inner, outer, upper lid, lower lid.
pub const RIGHT_EYE: [usize; 4] = [362, 263, 386, 374];
pub const SCORE_THRESH: f32 = 0.5;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GazeFeat {
    pub iris_nx: f32,
    pub iris_ny: f32,
    pub nose_nx: f32,
    pub nose_ny: f32,
    pub iris_l: bool,
    pub iris_r: bool,
    pub ok: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceRoi {
    pub cx: f32,
    pub cy: f32,
    pub size: f32,
    pub angle: f32,
}

/// Square from `max(bbox.w, bbox.h)`, scale 1.5× (margin 0.25). Rotate
/// from SCRFD 5 kps so the eye line is level; skip rotate when `kps` is None.
pub fn face_roi(face: &DetectedFace) -> FaceRoi {
    let side = face.bbox.w.max(face.bbox.h);
    let size = (side * (1.0 + 2.0 * ROI_MARGIN)).max(8.0);
    let angle = match face.kps {
        Some(kps) => {
            let (lx, ly) = kps[0];
            let (rx, ry) = kps[1];
            -(ry - ly).atan2(rx - lx)
        }
        None => 0.0,
    };
    FaceRoi {
        cx: face.cx(),
        cy: face.cy(),
        size,
        angle,
    }
}

pub fn roi_to_frame(roi: FaceRoi, crop: f32, x: f32, y: f32) -> (f32, f32) {
    let nx = x / crop - 0.5;
    let ny = y / crop - 0.5;
    let (sa, ca) = roi.angle.sin_cos();
    let rx = nx * roi.size;
    let ry = ny * roi.size;
    (roi.cx + rx * ca - ry * sa, roi.cy + rx * sa + ry * ca)
}

pub fn warp_roi_rgb(rgb: &[u8], w: u32, h: u32, roi: FaceRoi, out: usize) -> Vec<u8> {
    let mut dst = vec![0u8; out * out * 3];
    if w == 0 || h == 0 || roi.size <= 1.0 {
        return dst;
    }
    let (sa, ca) = roi.angle.sin_cos();
    for y in 0..out {
        for x in 0..out {
            let nx = (x as f32 + 0.5) / out as f32 - 0.5;
            let ny = (y as f32 + 0.5) / out as f32 - 0.5;
            let rx = nx * roi.size;
            let ry = ny * roi.size;
            let sx = roi.cx + rx * ca - ry * sa;
            let sy = roi.cy + rx * sa + ry * ca;
            if sx < 0.0 || sy < 0.0 || sx >= w as f32 || sy >= h as f32 {
                continue;
            }
            let ix = sx.floor() as u32;
            let iy = sy.floor() as u32;
            let i = ((iy * w + ix) * 3) as usize;
            let o = (y * out + x) * 3;
            if i + 2 < rgb.len() {
                dst[o] = rgb[i];
                dst[o + 1] = rgb[i + 1];
                dst[o + 2] = rgb[i + 2];
            }
        }
    }
    dst
}

/// Packed 478×3 has irises; 468×3 does not. Accepts NHWC or NCHW-last.
pub fn unpack_landmarks(data: &[f32]) -> Option<(Vec<(f32, f32, f32)>, bool)> {
    unpack_landmarks_shaped(&[], data)
}

pub fn unpack_landmarks_shaped(
    shape: &[i64],
    data: &[f32],
) -> Option<(Vec<(f32, f32, f32)>, bool)> {
    if data.is_empty() {
        return None;
    }
    let (n, has_iris, planar) = layout(shape, data.len())?;
    let mut pts = Vec::with_capacity(n);
    if planar {
        for i in 0..n {
            pts.push((data[i], data[n + i], data[2 * n + i]));
        }
    } else {
        for i in 0..n {
            pts.push((data[i * 3], data[i * 3 + 1], data[i * 3 + 2]));
        }
    }
    scale_if_unit(&mut pts);
    Some((pts, has_iris))
}

fn layout(shape: &[i64], len: usize) -> Option<(usize, bool, bool)> {
    let dims: Vec<usize> = shape
        .iter()
        .copied()
        .filter(|&d| d > 1)
        .map(|d| d as usize)
        .collect();
    if dims.len() >= 2 {
        let a = dims[dims.len() - 2];
        let b = dims[dims.len() - 1];
        if (a == 478 || a == 468) && b == 3 {
            return Some((a, a >= 478, false));
        }
        if a == 3 && (b == 478 || b == 468) {
            return Some((b, b >= 478, true));
        }
    }
    if len >= 478 * 3 {
        Some((478, true, false))
    } else if len >= 468 * 3 {
        Some((468, false, false))
    } else {
        None
    }
}

fn scale_if_unit(pts: &mut [(f32, f32, f32)]) {
    let max_xy = pts
        .iter()
        .map(|p| p.0.abs().max(p.1.abs()))
        .fold(0.0f32, f32::max);
    if max_xy <= 1.5 {
        let s = LANDMARKER_SIZE as f32;
        for p in pts {
            p.0 *= s;
            p.1 *= s;
        }
    }
}

/// Iris center in the eye box, clamped 0..1 in **camera** space (0 = left).
/// Selfie mirror is applied once in `gaze::uncalibrated`, not here.
pub fn iris_in_eye(iris: (f32, f32), corners: [(f32, f32); 4]) -> Option<(f32, f32)> {
    let min_x = corners.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let max_x = corners
        .iter()
        .map(|p| p.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = corners.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let max_y = corners
        .iter()
        .map(|p| p.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let dw = max_x - min_x;
    let dh = max_y - min_y;
    if dw < 1e-3 || dh < 1e-3 {
        return None;
    }
    let nx = ((iris.0 - min_x) / dw).clamp(0.0, 1.0);
    let ny = ((iris.1 - min_y) / dh).clamp(0.0, 1.0);
    Some((nx, ny))
}

pub fn feat_from_landmarks(
    pts: &[(f32, f32, f32)],
    has_iris: bool,
    roi: FaceRoi,
    crop: f32,
    frame_w: f32,
    frame_h: f32,
) -> GazeFeat {
    let to_frame = |i: usize| -> Option<(f32, f32)> {
        let p = pts.get(i)?;
        Some(roi_to_frame(roi, crop, p.0, p.1))
    };
    let corners = |idx: [usize; 4]| -> Option<[(f32, f32); 4]> {
        Some([
            to_frame(idx[0])?,
            to_frame(idx[1])?,
            to_frame(idx[2])?,
            to_frame(idx[3])?,
        ])
    };
    let mut feat = GazeFeat::default();
    if let Some(n) = to_frame(NOSE_TIP) {
        if frame_w > 1.0 && frame_h > 1.0 {
            feat.nose_nx = (n.0 / frame_w).clamp(0.0, 1.0);
            feat.nose_ny = (n.1 / frame_h).clamp(0.0, 1.0);
        }
    }
    if !has_iris {
        return feat;
    }
    let mut nxs = 0.0;
    let mut nys = 0.0;
    let mut n = 0u32;
    if let (Some(iris), Some(boxc)) = (to_frame(IRIS_L), corners(LEFT_EYE)) {
        if let Some((nx, ny)) = iris_in_eye(iris, boxc) {
            nxs += nx;
            nys += ny;
            n += 1;
            feat.iris_l = true;
        }
    }
    if let (Some(iris), Some(boxc)) = (to_frame(IRIS_R), corners(RIGHT_EYE)) {
        if let Some((nx, ny)) = iris_in_eye(iris, boxc) {
            nxs += nx;
            nys += ny;
            n += 1;
            feat.iris_r = true;
        }
    }
    if n > 0 {
        feat.iris_nx = nxs / n as f32;
        feat.iris_ny = nys / n as f32;
        feat.ok = true;
    }
    feat
}

/// Run the landmarker when the ONNX file is present. `None` = no net.
pub fn infer(rgb: &[u8], w: u32, h: u32, face: &DetectedFace) -> Option<GazeFeat> {
    onnx::infer(rgb, w, h, face)
}

fn landmarker_path() -> Option<std::path::PathBuf> {
    let dir = crate::identity::models_dir()?;
    let p = dir.join("face_landmarker.onnx");
    p.is_file().then_some(p)
}

mod onnx {
    use super::*;
    use ndarray::Array4;
    use std::sync::Mutex;

    struct Net {
        session: ort::session::Session,
        input: String,
        broken: bool,
    }

    static SESS: Mutex<Option<Net>> = Mutex::new(None);
    static LOGGED: Mutex<bool> = Mutex::new(false);

    fn session() -> Option<()> {
        let mut g = SESS.lock().ok()?;
        if g.is_some() {
            return Some(());
        }
        let path = super::landmarker_path()?;
        let sess = crate::identity::ort_sess::session_from_file(&path)?;
        let input = sess.inputs().first()?.name().to_string();
        if let Ok(mut logged) = LOGGED.lock() {
            if !*logged {
                eprintln!("buckyboi: face landmarker 256×256 (478 iris) input={input}");
                *logged = true;
            }
        }
        *g = Some(Net {
            session: sess,
            input,
            broken: false,
        });
        Some(())
    }

    fn rgb_blob(rgb: &[u8], w: u32, h: u32) -> Array4<f32> {
        let size = LANDMARKER_SIZE as usize;
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
                blob[[0, 0, y, x]] = rgb[i] as f32 / 255.0;
                blob[[0, 1, y, x]] = rgb[i + 1] as f32 / 255.0;
                blob[[0, 2, y, x]] = rgb[i + 2] as f32 / 255.0;
            }
        }
        blob
    }

    fn as_prob(x: f32) -> f32 {
        if x > 0.0 && x < 1.0 {
            x
        } else {
            1.0 / (1.0 + (-x).exp())
        }
    }

    pub fn infer(rgb: &[u8], w: u32, h: u32, face: &DetectedFace) -> Option<GazeFeat> {
        session()?;
        let mut g = SESS.lock().ok()?;
        let net = g.as_mut()?;
        if net.broken {
            return None;
        }
        let roi = face_roi(face);
        let crop = warp_roi_rgb(rgb, w, h, roi, LANDMARKER_SIZE as usize);
        let blob = rgb_blob(&crop, LANDMARKER_SIZE, LANDMARKER_SIZE);
        let input = ort::value::Tensor::from_array(blob).ok()?;
        let name = net.input.clone();
        let outputs = match net.session.run(ort::inputs![name.as_str() => input]) {
            Ok(o) => o,
            Err(e) => {
                if !net.broken {
                    eprintln!(
                        "buckyboi: face landmarker failed ({e}) — gaze falls back to face-box"
                    );
                    net.broken = true;
                }
                return None;
            }
        };
        let mut best: Option<(Vec<(f32, f32, f32)>, bool)> = None;
        let mut score_ok = true;
        for (_, t) in outputs.iter() {
            let Ok((shape, data)) = t.try_extract_tensor::<f32>() else {
                continue;
            };
            let data = data.to_vec();
            if data.len() <= 4 {
                if let Some(&s) = data.first() {
                    score_ok = as_prob(s) >= SCORE_THRESH;
                }
                continue;
            }
            let shape: Vec<i64> = shape.iter().map(|d| *d as i64).collect();
            if let Some(lm) = unpack_landmarks_shaped(&shape, &data) {
                let prefer = lm.1;
                if best.as_ref().map(|b| b.1).unwrap_or(false) && !prefer {
                    continue;
                }
                best = Some(lm);
            }
        }
        if !score_ok {
            return Some(GazeFeat::default());
        }
        let (pts, has_iris) = best?;
        Some(feat_from_landmarks(
            &pts,
            has_iris,
            roi,
            LANDMARKER_SIZE as f32,
            w as f32,
            h as f32,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::scrfd::{DetectedFace, FaceBox};

    fn ramp(n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n * 3];
        for i in 0..n {
            v[i * 3] = i as f32;
            v[i * 3 + 1] = i as f32 + 0.5;
            v[i * 3 + 2] = 1.0;
        }
        v
    }

    #[test]
    fn canned_478_has_irises() {
        let (pts, has) = unpack_landmarks(&ramp(478)).expect("478");
        assert!(has);
        assert_eq!(pts.len(), 478);
        assert!((pts[IRIS_L].0 - 468.0).abs() < 1e-4);
        assert!((pts[IRIS_R].0 - 473.0).abs() < 1e-4);
        assert!((pts[NOSE_TIP].0 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn canned_468_has_no_irises() {
        let (pts, has) = unpack_landmarks(&ramp(468)).expect("468");
        assert!(!has);
        assert_eq!(pts.len(), 468);
        assert!(pts.get(IRIS_L).is_none());
    }

    #[test]
    fn iris_in_eye_uses_mesh_corners() {
        // Left eye {33,133,159,145}: outer, inner, upper, lower.
        let corners = [(10.0, 20.0), (30.0, 20.0), (20.0, 10.0), (20.0, 30.0)];
        let (nx, ny) = iris_in_eye((20.0, 20.0), corners).unwrap();
        assert!((nx - 0.5).abs() < 1e-4);
        assert!((ny - 0.5).abs() < 1e-4);
        let (nx, _) = iris_in_eye((10.0, 20.0), corners).unwrap();
        // Camera-left of the eye box stays left (no pre-mirror).
        assert!((nx - 0.0).abs() < 1e-4);
        assert_eq!(LEFT_EYE, [33, 133, 159, 145]);
        assert_eq!(RIGHT_EYE, [362, 263, 386, 374]);
        let (sx, _) = crate::gaze::uncalibrated(nx, 0.5, 0.5, 0.5, 1920.0, 1080.0);
        let (face_sx, _) =
            crate::gaze::face_to_screen(10.0, 20.0, 80.0, 40.0, 1920.0, 1080.0, true);
        assert!(
            sx > 960.0 && face_sx > 960.0,
            "camera-left iris {sx} vs face_to_screen {face_sx}"
        );
    }

    #[test]
    fn roi_margin_and_no_rotate_without_kps() {
        let face = DetectedFace {
            bbox: FaceBox {
                x: 10.0,
                y: 20.0,
                w: 40.0,
                h: 80.0,
            },
            kps: None,
            score: 0.9,
        };
        let roi = face_roi(&face);
        assert!((roi.size - 80.0 * 1.5).abs() < 1e-3);
        assert_eq!(roi.angle, 0.0);
        assert!((roi.cx - 30.0).abs() < 1e-3);
        assert!((roi.cy - 60.0).abs() < 1e-3);
        let mut kps = [(0.0, 0.0); 5];
        kps[0] = (0.0, 10.0);
        kps[1] = (10.0, 10.0);
        let mut f2 = face;
        f2.kps = Some(kps);
        assert!((face_roi(&f2).angle).abs() < 1e-4);
        kps[1] = (10.0, 20.0);
        f2.kps = Some(kps);
        assert!(face_roi(&f2).angle.abs() > 0.1);
    }

    #[test]
    fn feat_averages_both_irises() {
        let mut pts = vec![(0.0, 0.0, 0.0); 478];
        pts[33] = (10.0, 20.0, 0.0);
        pts[133] = (30.0, 20.0, 0.0);
        pts[159] = (20.0, 10.0, 0.0);
        pts[145] = (20.0, 30.0, 0.0);
        pts[IRIS_L] = (20.0, 20.0, 0.0);
        pts[362] = (50.0, 20.0, 0.0);
        pts[263] = (70.0, 20.0, 0.0);
        pts[386] = (60.0, 10.0, 0.0);
        pts[374] = (60.0, 30.0, 0.0);
        pts[IRIS_R] = (60.0, 20.0, 0.0);
        pts[NOSE_TIP] = (40.0, 40.0, 0.0);
        // Identity ROI: crop pixels map 1:1 into the 256 frame.
        let roi = FaceRoi {
            cx: 128.0,
            cy: 128.0,
            size: 256.0,
            angle: 0.0,
        };
        let feat = feat_from_landmarks(&pts, true, roi, 256.0, 256.0, 256.0);
        assert!(feat.ok && feat.iris_l && feat.iris_r);
        assert!((feat.iris_nx - 0.5).abs() < 0.05);
        let no = feat_from_landmarks(&pts[..468], false, roi, 256.0, 256.0, 256.0);
        assert!(!no.ok && !no.iris_l && !no.iris_r);
    }

    #[test]
    fn arcface_r50_session_probe_if_present() {
        let dir = crate::identity::models_dir().unwrap();
        let path = dir.join("w600k_r50.onnx");
        if !path.is_file() {
            return;
        }
        use ndarray::Array4;
        use ort::session::builder::GraphOptimizationLevel;
        use ort::session::Session;
        let blob = Array4::<f32>::zeros((1, 3, 112, 112));
        for (label, level) in [
            ("disable", GraphOptimizationLevel::Disable),
            ("l1", GraphOptimizationLevel::Level1),
        ] {
            let mut sess = Session::builder()
                .unwrap()
                .with_optimization_level(level)
                .unwrap()
                .with_intra_threads(1)
                .unwrap()
                .commit_from_file(&path)
                .unwrap();
            eprintln!(
                "r50 {label} inputs {:?}",
                sess.inputs()
                    .iter()
                    .map(|i| format!("{} {:?}", i.name(), i.dtype()))
                    .collect::<Vec<_>>()
            );
            let t = ort::value::Tensor::from_array(blob.clone()).unwrap();
            let name = sess.inputs()[0].name().to_string();
            let msg = match sess.run(ort::inputs![name.as_str() => t]) {
                Ok(_) => format!("r50 {label} named run ok"),
                Err(e) => format!("r50 {label} named run err {e}"),
            };
            eprintln!("{msg}");
        }
    }

    #[test]
    fn landmarker_session_probe_if_present() {
        let Some(path) = super::landmarker_path() else {
            return;
        };
        use ndarray::Array4;
        use ort::session::builder::GraphOptimizationLevel;
        use ort::session::Session;
        let blob = Array4::<f32>::zeros((1, 3, 256, 256));
        for (label, level) in [
            ("disable", GraphOptimizationLevel::Disable),
            ("l1", GraphOptimizationLevel::Level1),
        ] {
            let mut sess = Session::builder()
                .unwrap()
                .with_optimization_level(level)
                .unwrap()
                .with_intra_threads(1)
                .unwrap()
                .commit_from_file(&path)
                .unwrap();
            eprintln!(
                "landmarker {label} inputs {:?}",
                sess.inputs()
                    .iter()
                    .map(|i| format!("{} {:?}", i.name(), i.dtype()))
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "landmarker {label} outputs {:?}",
                sess.outputs()
                    .iter()
                    .map(|o| format!("{} {:?}", o.name(), o.dtype()))
                    .collect::<Vec<_>>()
            );
            let t = ort::value::Tensor::from_array(blob.clone()).unwrap();
            let name = sess.inputs()[0].name().to_string();
            let msg = match sess.run(ort::inputs![name.as_str() => t]) {
                Ok(_) => format!("landmarker {label} named run ok"),
                Err(e) => format!("landmarker {label} named run err {e}"),
            };
            eprintln!("{msg}");
            let t2 = ort::value::Tensor::from_array(blob.clone()).unwrap();
            let msg = match sess.run(ort::inputs![t2]) {
                Ok(_) => format!("landmarker {label} positional run ok"),
                Err(e) => format!("landmarker {label} positional run err {e}"),
            };
            eprintln!("{msg}");
        }
    }
}
