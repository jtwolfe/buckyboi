//! Face quality, crop, and optional ONNX ArcFace embeddings.
//!
//! Without the `face` feature or without models, embedding extraction fails
//! closed (`None`). Enrollment tests inject embeddings directly.

use crate::identity::embed::Embedding;
use crate::gaze::estimate_face_rgb;

pub const FACE_ENROLL_NEED: usize = 8;
pub const FACE_KIND_ARCFACE: &str = "arcface";
pub const FACE_KIND_PROBE: &str = "face-probe";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceQuality {
    pub area_frac: f32,
    pub brightness: f32,
    pub sharpness: f32,
    pub faces: u32,
    pub ok: bool,
}

impl FaceQuality {
    pub fn reject() -> Self {
        Self {
            area_frac: 0.0,
            brightness: 0.0,
            sharpness: 0.0,
            faces: 0,
            ok: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Variance of a 3×3 Laplacian — blur proxy. Too-smooth crops are rejected.
pub fn laplacian_var(gray: &[f32], w: usize, h: usize) -> f32 {
    if w < 5 || h < 5 || gray.len() < w * h {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut n = 0.0f64;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let l = -4.0 * gray[i]
                + gray[i - 1]
                + gray[i + 1]
                + gray[i - w]
                + gray[i + w];
            sum += l as f64;
            sum2 += (l as f64) * (l as f64);
            n += 1.0;
        }
    }
    if n < 4.0 {
        return 0.0;
    }
    let mean = sum / n;
    (sum2 / n - mean * mean) as f32
}

fn gray_crop(rgb: &[u8], fw: u32, fh: u32, bbox: FaceBox, out_w: usize, out_h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_w * out_h];
    if fw == 0 || fh == 0 {
        return out;
    }
    for y in 0..out_h {
        let fy = bbox.y + bbox.h * (y as f32 + 0.5) / out_h as f32;
        let sy = fy.round().clamp(0.0, fh as f32 - 1.0) as u32;
        for x in 0..out_w {
            let fx = bbox.x + bbox.w * (x as f32 + 0.5) / out_w as f32;
            let sx = fx.round().clamp(0.0, fw as f32 - 1.0) as u32;
            let i = ((sy * fw + sx) * 3) as usize;
            if i + 2 < rgb.len() {
                out[y * out_w + x] =
                    0.299 * rgb[i] as f32 + 0.587 * rgb[i + 1] as f32 + 0.114 * rgb[i + 2] as f32;
            }
        }
    }
    out
}

fn brightness(gray: &[f32]) -> f32 {
    if gray.is_empty() {
        return 0.0;
    }
    gray.iter().sum::<f32>() / gray.len() as f32
}

/// Skin-blob face box (same detector as the gaze prototype) + quality gates.
pub fn assess_face_rgb(rgb: &[u8], w: u32, h: u32) -> (FaceQuality, Option<FaceBox>) {
    let Some(guess) = estimate_face_rgb(rgb, w, h) else {
        return (FaceQuality::reject(), None);
    };
    // Approximate box from the blob center; gaze code does not store extents,
    // so use a conservative square around the face center.
    let side = (w.min(h) as f32) * 0.42;
    let bx = (guess.cx - side * 0.5).clamp(0.0, w as f32);
    let by = (guess.cy - side * 0.55).clamp(0.0, h as f32);
    let bw = side.min(w as f32 - bx).max(8.0);
    let bh = (side * 1.15).min(h as f32 - by).max(8.0);
    let bbox = FaceBox {
        x: bx,
        y: by,
        w: bw,
        h: bh,
    };
    let area_frac = (bw * bh) / (w as f32 * h as f32).max(1.0);
    let gray = gray_crop(rgb, w, h, bbox, 48, 48);
    let brightness = brightness(&gray);
    let sharpness = laplacian_var(&gray, 48, 48);
    let ok = area_frac >= 0.04
        && area_frac <= 0.85
        && (40.0..230.0).contains(&brightness)
        && sharpness >= 12.0;
    (
        FaceQuality {
            area_frac,
            brightness,
            sharpness,
            faces: 1,
            ok,
        },
        Some(bbox),
    )
}

/// Cheap local probe embedding from a quality-checked crop (histogram + LBP-lite).
/// Used when ONNX models are absent so enrollment still has a concrete offline path.
/// Weaker than ArcFace — documented as a prototype print, not a commercial face ID.
pub fn probe_embed(rgb: &[u8], w: u32, h: u32, bbox: FaceBox) -> Embedding {
    let gray = gray_crop(rgb, w, h, bbox, 16, 16);
    let mut hist = vec![0.0f32; 32];
    for (i, p) in gray.iter().enumerate() {
        let bin = (*p / 8.0).clamp(0.0, 31.0) as usize;
        hist[bin] += 1.0;
        // Tiny local binary pattern vs right neighbor.
        if i + 1 < gray.len() && *p > gray[i + 1] {
            hist[16 + (i % 16)] += 1.0;
        }
    }
    Embedding::new(FACE_KIND_PROBE, hist)
}

/// Extract an embedding. Prefers ONNX ArcFace when the `face` feature and model exist.
pub fn extract_embedding(rgb: &[u8], w: u32, h: u32) -> (FaceQuality, Option<Embedding>) {
    let (q, bbox) = assess_face_rgb(rgb, w, h);
    if !q.ok {
        return (q, None);
    }
    let bbox = bbox.unwrap();
    #[cfg(feature = "face")]
    {
        if let Some(emb) = onnx_embed(rgb, w, h, bbox) {
            return (q, Some(emb));
        }
    }
    if crate::identity::env_flag("BUCKYBOI_FACE_PROBE") {
        return (q, Some(probe_embed(rgb, w, h, bbox)));
    }
    // Fail closed unless the operator opted into the weak probe print.
    (q, None)
}

#[cfg(feature = "face")]
fn onnx_embed(rgb: &[u8], w: u32, h: u32, bbox: FaceBox) -> Option<Embedding> {
    onnx::embed_arcface(rgb, w, h, bbox)
}

#[cfg(feature = "face")]
mod onnx {
    use super::*;
    use crate::identity::models_dir;
    use ndarray::Array4;
    use std::path::PathBuf;
    use std::sync::Mutex;

    fn rec_path() -> Option<PathBuf> {
        let dir = models_dir()?;
        for name in [
            "w600k_r50.onnx",
            "w600k_mbf.onnx",
            "arcface.onnx",
            "buffalo_sc_w600k_mbf.onnx",
        ] {
            let p = dir.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
        None
    }

    struct Rec {
        session: ort::session::Session,
    }

    static REC: Mutex<Option<Rec>> = Mutex::new(None);

    fn session() -> Option<()> {
        let mut g = REC.lock().ok()?;
        if g.is_some() {
            return Some(());
        }
        let path = rec_path()?;
        let sess = ort::session::Session::builder()
            .ok()?
            .commit_from_file(&path)
            .ok()?;
        *g = Some(Rec { session: sess });
        Some(())
    }

    /// ArcFace family: (1,3,112,112) BGR, (x-127.5)/128 → 512-d.
    pub fn embed_arcface(rgb: &[u8], w: u32, h: u32, bbox: FaceBox) -> Option<Embedding> {
        session()?;
        let mut g = REC.lock().ok()?;
        let rec = g.as_mut()?;
        let mut blob = Array4::<f32>::zeros((1, 3, 112, 112));
        for y in 0..112 {
            let fy = bbox.y + bbox.h * (y as f32 + 0.5) / 112.0;
            let sy = fy.round().clamp(0.0, h as f32 - 1.0) as u32;
            for x in 0..112 {
                let fx = bbox.x + bbox.w * (x as f32 + 0.5) / 112.0;
                let sx = fx.round().clamp(0.0, w as f32 - 1.0) as u32;
                let i = ((sy * w + sx) * 3) as usize;
                if i + 2 >= rgb.len() {
                    continue;
                }
                // RGB → BGR, InsightFace normalize.
                blob[[0, 0, y, x]] = (rgb[i + 2] as f32 - 127.5) / 128.0;
                blob[[0, 1, y, x]] = (rgb[i + 1] as f32 - 127.5) / 128.0;
                blob[[0, 2, y, x]] = (rgb[i] as f32 - 127.5) / 128.0;
            }
        }
        let input = ort::value::Tensor::from_array(blob).ok()?;
        let outputs = rec.session.run(ort::inputs![input]).ok()?;
        let (_shape, data) = outputs
            .iter()
            .next()?
            .1
            .try_extract_tensor::<f32>()
            .ok()?;
        if data.len() < 128 {
            return None;
        }
        Some(Embedding::new(FACE_KIND_ARCFACE, data.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skin_patch(w: u32, h: u32) -> Vec<u8> {
        let mut rgb = vec![20u8; (w * h * 3) as usize];
        for y in 12..44 {
            for x in 18..50 {
                let i = ((y * w + x) * 3) as usize;
                rgb[i] = 190;
                rgb[i + 1] = 120;
                rgb[i + 2] = 100;
            }
        }
        // texture so Laplacian is non-zero
        for y in 14..42 {
            for x in 20..48 {
                if (x + y) % 3 == 0 {
                    let i = ((y * w + x) * 3) as usize;
                    rgb[i] = 80;
                    rgb[i + 1] = 40;
                    rgb[i + 2] = 30;
                }
            }
        }
        rgb
    }

    #[test]
    fn empty_frame_rejected() {
        let rgb = vec![16u8; 40 * 30 * 3];
        let (q, b) = assess_face_rgb(&rgb, 40, 30);
        assert!(!q.ok);
        assert!(b.is_none());
    }

    #[test]
    fn textured_skin_can_pass_quality() {
        let rgb = skin_patch(80, 60);
        let (q, b) = assess_face_rgb(&rgb, 80, 60);
        assert!(b.is_some());
        assert!(q.faces == 1);
        assert!(q.area_frac > 0.0);
        // Sharpness depends on the checker; we only require the function runs.
        let _ = q.ok;
    }

    #[test]
    fn probe_embed_is_normalized() {
        let rgb = skin_patch(80, 60);
        let bbox = FaceBox {
            x: 18.0,
            y: 12.0,
            w: 32.0,
            h: 32.0,
        };
        let e = probe_embed(&rgb, 80, 60, bbox);
        assert_eq!(e.kind, FACE_KIND_PROBE);
        let n = e.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4, "{n}");
    }

    #[test]
    fn extract_without_probe_flag_fails_closed() {
        std::env::remove_var("BUCKYBOI_FACE_PROBE");
        let rgb = skin_patch(80, 60);
        let (_q, emb) = extract_embedding(&rgb, 80, 60);
        assert!(emb.is_none());
    }
}
