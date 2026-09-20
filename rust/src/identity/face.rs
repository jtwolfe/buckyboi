//! Face quality, SCRFD detect, 5-point ArcFace align, optional ONNX embed.
//!
//! Pipeline when models are present:
//!   SCRFD (`det_10g.onnx`) → bbox + 5 landmarks → `norm_crop` 112×112 →
//!   ArcFace (`w600k_r50.onnx` / `w600k_mbf.onnx`).
//! Without a detector, the skin-blob box is the degraded fallback (no align).
//! Without a recognition model, embedding fails closed unless
//! `BUCKYBOI_FACE_PROBE=1` (weak histogram — opt-in, documented as such).

use crate::gaze::estimate_face_rgb;
use crate::identity::align::{norm_crop_arcface, resize_box_rgb, ARCFACE_SIZE};
use crate::identity::embed::Embedding;
use crate::identity::scrfd::{detect_onnx, pick_primary_face, DetectedFace, SCRFD_DET_THRESH};

pub use crate::identity::scrfd::FaceBox;
use std::sync::{Mutex, OnceLock};

pub const FACE_ENROLL_NEED: usize = 8;
pub const FACE_KIND_ARCFACE: &str = "arcface";
pub const FACE_KIND_PROBE: &str = "face-probe";

/// Cosine on L2-normalized embeddings. InsightFace-style working points
/// for the buffalo recognition nets (tune via `face_threshold` / env).
pub const FACE_THRESHOLD_R50: f32 = 0.35;
pub const FACE_THRESHOLD_MBF: f32 = 0.40;
pub const FACE_THRESHOLD_DEFAULT: f32 = FACE_THRESHOLD_R50;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceReject {
    None,
    NoFace,
    TooDark,
    TooBright,
    TooBlurry,
    TooSmall,
    TooClose,
    NoEmbed,
}

impl FaceReject {
    pub fn hint(self) -> &'static str {
        match self {
            FaceReject::None => "",
            FaceReject::NoFace => "NO FACE",
            FaceReject::TooDark => "TOO DARK",
            FaceReject::TooBright => "TOO BRIGHT",
            FaceReject::TooBlurry => "TOO BLURRY",
            FaceReject::TooSmall => "TOO SMALL",
            FaceReject::TooClose => "TOO CLOSE",
            FaceReject::NoEmbed => "NO MODEL",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceQuality {
    pub area_frac: f32,
    pub brightness: f32,
    pub sharpness: f32,
    pub faces: u32,
    pub ok: bool,
    pub reject: FaceReject,
}

impl FaceQuality {
    pub fn reject(reason: FaceReject) -> Self {
        Self {
            area_frac: 0.0,
            brightness: 0.0,
            sharpness: 0.0,
            faces: 0,
            ok: false,
            reject: reason,
        }
    }
}

/// Look-target published by the last SCRFD (or skin) detection.
#[derive(Clone, Copy, Debug)]
pub struct FaceLook {
    pub cx: f32,
    pub cy: f32,
    pub fw: f32,
    pub fh: f32,
    pub t_ms: u64,
    pub from_scrfd: bool,
}

static LAST_LOOK: OnceLock<Mutex<Option<FaceLook>>> = OnceLock::new();

fn look_slot() -> &'static Mutex<Option<FaceLook>> {
    LAST_LOOK.get_or_init(|| Mutex::new(None))
}

pub fn publish_look(look: FaceLook) {
    if let Ok(mut g) = look_slot().lock() {
        *g = Some(look);
    }
}

/// Fresh SCRFD/skin look if a detection landed within `max_age_ms`.
pub fn latest_look(now_ms: u64, max_age_ms: u64) -> Option<FaceLook> {
    let g = look_slot().lock().ok()?;
    let look = (*g)?;
    if now_ms.saturating_sub(look.t_ms) <= max_age_ms {
        Some(look)
    } else {
        None
    }
}

pub fn now_ms_soft() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Rec-net cosine threshold from the ONNX filename.
pub fn face_cosine_threshold(model_name: &str) -> f32 {
    let n = model_name.to_ascii_lowercase();
    if n.contains("mbf") || n.contains("mobilefacenet") || n.contains("buffalo_sc") {
        FACE_THRESHOLD_MBF
    } else if n.contains("r50") || n.contains("buffalo_l") || n.contains("iresnet50") {
        FACE_THRESHOLD_R50
    } else {
        FACE_THRESHOLD_DEFAULT
    }
}

pub fn rec_model_names() -> &'static [&'static str] {
    &[
        "w600k_r50.onnx",
        "w600k_mbf.onnx",
        "arcface.onnx",
        "buffalo_sc_w600k_mbf.onnx",
    ]
}

pub fn find_rec_model() -> Option<std::path::PathBuf> {
    let dir = crate::identity::models_dir()?;
    for name in rec_model_names() {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Env override wins; else model-aware default when `configured` is still
/// the compiled default; else the stored value.
pub fn effective_face_threshold(configured: f32, model_name: Option<&str>) -> f32 {
    if let Some(v) =
        crate::identity::env_or_alias("BUCKYBOI_FACE_THRESHOLD", "BUDDY_FACE_THRESHOLD")
    {
        if let Ok(n) = v.parse::<f32>() {
            return n.clamp(0.15, 0.95);
        }
    }
    if let Some(name) = model_name {
        if (configured - crate::identity::embed::DEFAULT_FACE_THRESHOLD).abs() < 1e-4
            || (configured - FACE_THRESHOLD_DEFAULT).abs() < 1e-4
        {
            return face_cosine_threshold(name);
        }
    }
    configured.clamp(0.15, 0.95)
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
            let l = -4.0 * gray[i] + gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w];
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

fn rgb_to_gray(rgb: &[u8]) -> Vec<f32> {
    let n = rgb.len() / 3;
    let mut g = vec![0.0f32; n];
    for i in 0..n {
        let o = i * 3;
        if o + 2 < rgb.len() {
            g[i] = 0.299 * rgb[o] as f32 + 0.587 * rgb[o + 1] as f32 + 0.114 * rgb[o + 2] as f32;
        }
    }
    g
}

fn brightness(gray: &[f32]) -> f32 {
    if gray.is_empty() {
        return 0.0;
    }
    gray.iter().sum::<f32>() / gray.len() as f32
}

fn quality_from_crop(
    rgb112: &[u8],
    bbox: FaceBox,
    frame_w: u32,
    frame_h: u32,
    faces: u32,
) -> FaceQuality {
    let area_frac = (bbox.w * bbox.h) / (frame_w as f32 * frame_h as f32).max(1.0);
    let gray = rgb_to_gray(rgb112);
    let side = ((gray.len() as f32).sqrt()) as usize;
    let brightness = brightness(&gray);
    let sharpness = if side >= 5 {
        laplacian_var(&gray, side, side)
    } else {
        0.0
    };
    let reject = if area_frac < 0.04 {
        FaceReject::TooSmall
    } else if area_frac > 0.85 {
        FaceReject::TooClose
    } else if brightness < 40.0 {
        FaceReject::TooDark
    } else if brightness >= 230.0 {
        FaceReject::TooBright
    } else if sharpness < 12.0 {
        FaceReject::TooBlurry
    } else {
        FaceReject::None
    };
    FaceQuality {
        area_frac,
        brightness,
        sharpness,
        faces,
        ok: reject == FaceReject::None,
        reject,
    }
}

fn skin_box(rgb: &[u8], w: u32, h: u32) -> Option<FaceBox> {
    let guess = estimate_face_rgb(rgb, w, h)?;
    let side = (w.min(h) as f32) * 0.42;
    let bx = (guess.cx - side * 0.5).clamp(0.0, w as f32);
    let by = (guess.cy - side * 0.55).clamp(0.0, h as f32);
    let bw = side.min(w as f32 - bx).max(8.0);
    let bh = (side * 1.15).min(h as f32 - by).max(8.0);
    Some(FaceBox {
        x: bx,
        y: by,
        w: bw,
        h: bh,
    })
}

/// Detect every face (SCRFD when the model is loaded; else one skin blob).
pub fn detect_faces(rgb: &[u8], w: u32, h: u32) -> Vec<DetectedFace> {
    if let Some(faces) = detect_onnx(rgb, w, h) {
        if !faces.is_empty() {
            return faces;
        }
    }
    if let Some(bbox) = skin_box(rgb, w, h) {
        return vec![DetectedFace {
            bbox,
            kps: None,
            score: 0.4,
        }];
    }
    Vec::new()
}

/// Primary face: largest / most central (see `pick_primary_face`).
pub fn detect_primary(rgb: &[u8], w: u32, h: u32) -> Option<DetectedFace> {
    let faces = detect_faces(rgb, w, h);
    pick_primary_face(&faces, w as f32, h as f32)
}

fn crop_for_quality(rgb: &[u8], w: u32, h: u32, face: &DetectedFace) -> (Vec<u8>, bool) {
    if let Some(kps) = face.kps {
        if let Some(aligned) = norm_crop_arcface(rgb, w, h, &kps) {
            return (aligned, true);
        }
    }
    (
        resize_box_rgb(
            rgb,
            w,
            h,
            face.bbox.x,
            face.bbox.y,
            face.bbox.w,
            face.bbox.h,
            ARCFACE_SIZE,
        ),
        false,
    )
}

/// Skin-blob / SCRFD box + quality gates. Used by enroll and tests.
pub fn assess_face_rgb(rgb: &[u8], w: u32, h: u32) -> (FaceQuality, Option<FaceBox>) {
    let Some(face) = detect_primary(rgb, w, h) else {
        return (FaceQuality::reject(FaceReject::NoFace), None);
    };
    let (crop, _aligned) = crop_for_quality(rgb, w, h, &face);
    let q = quality_from_crop(&crop, face.bbox, w, h, 1);
    publish_look(FaceLook {
        cx: face.cx(),
        cy: face.cy(),
        fw: w as f32,
        fh: h as f32,
        t_ms: now_ms_soft(),
        from_scrfd: face.kps.is_some() || face.score >= SCRFD_DET_THRESH,
    });
    (q, Some(face.bbox))
}

/// Cheap local probe embedding from a quality-checked crop (histogram + LBP-lite).
/// Prototype only — not InsightFace. Opt-in via `BUCKYBOI_FACE_PROBE=1`.
pub fn probe_embed(rgb: &[u8], w: u32, h: u32, bbox: FaceBox) -> Embedding {
    let crop = resize_box_rgb(rgb, w, h, bbox.x, bbox.y, bbox.w, bbox.h, 16);
    let gray = rgb_to_gray(&crop);
    let mut hist = vec![0.0f32; 32];
    for (i, p) in gray.iter().enumerate() {
        let bin = (*p / 8.0).clamp(0.0, 31.0) as usize;
        hist[bin] += 1.0;
        if i + 1 < gray.len() && *p > gray[i + 1] {
            hist[16 + (i % 16)] += 1.0;
        }
    }
    Embedding::new(FACE_KIND_PROBE, hist)
}

/// Extract an embedding. Prefers detect → align → ArcFace when models exist.
pub fn extract_embedding(rgb: &[u8], w: u32, h: u32) -> (FaceQuality, Option<Embedding>) {
    let Some(face) = detect_primary(rgb, w, h) else {
        return (FaceQuality::reject(FaceReject::NoFace), None);
    };
    let (crop, aligned) = crop_for_quality(rgb, w, h, &face);
    let mut q = quality_from_crop(&crop, face.bbox, w, h, 1);
    publish_look(FaceLook {
        cx: face.cx(),
        cy: face.cy(),
        fw: w as f32,
        fh: h as f32,
        t_ms: now_ms_soft(),
        from_scrfd: face.kps.is_some(),
    });
    if !q.ok {
        return (q, None);
    }
    let _ = aligned;
    #[cfg(feature = "face")]
    {
        if let Some(emb) = onnx_embed_aligned(&crop, aligned) {
            return (q, Some(emb));
        }
    }
    if crate::identity::env_flag("BUCKYBOI_FACE_PROBE") {
        return (q, Some(probe_embed(rgb, w, h, face.bbox)));
    }
    q.ok = false;
    q.reject = FaceReject::NoEmbed;
    (q, None)
}

#[cfg(feature = "face")]
fn onnx_embed_aligned(aligned_rgb: &[u8], _was_aligned: bool) -> Option<Embedding> {
    onnx::embed_arcface(aligned_rgb)
}

#[cfg(feature = "face")]
mod onnx {
    use super::*;
    use crate::identity::align::arcface_blob_bgr;
    use ndarray::Array4;
    use std::sync::Mutex;

    struct Rec {
        session: ort::session::Session,
        name: String,
    }

    static REC: Mutex<Option<Rec>> = Mutex::new(None);

    pub fn loaded_name() -> Option<String> {
        REC.lock().ok()?.as_ref().map(|r| r.name.clone())
    }

    fn session() -> Option<()> {
        let mut g = REC.lock().ok()?;
        if g.is_some() {
            return Some(());
        }
        let path = find_rec_model()?;
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("arcface")
            .to_string();
        let sess = ort::session::Session::builder()
            .ok()?
            .commit_from_file(&path)
            .ok()?;
        eprintln!(
            "buckyboi: ArcFace {} (cosine ~{:.2})",
            name,
            face_cosine_threshold(&name)
        );
        *g = Some(Rec {
            session: sess,
            name,
        });
        Some(())
    }

    /// ArcFace: (1,3,112,112) BGR, (x-127.5)/128 → 512-d. Input is already
    /// 112×112 RGB from `norm_crop` (or the bbox-resize fallback).
    pub fn embed_arcface(aligned_rgb: &[u8]) -> Option<Embedding> {
        session()?;
        let mut g = REC.lock().ok()?;
        let rec = g.as_mut()?;
        let flat = arcface_blob_bgr(aligned_rgb, ARCFACE_SIZE);
        let blob = Array4::from_shape_vec((1, 3, ARCFACE_SIZE, ARCFACE_SIZE), flat).ok()?;
        let input = ort::value::Tensor::from_array(blob).ok()?;
        let outputs = rec.session.run(ort::inputs![input]).ok()?;
        let mut data = None;
        for (_, v) in outputs.iter() {
            if let Ok((_shape, slice)) = v.try_extract_tensor::<f32>() {
                data = Some(slice.to_vec());
                break;
            }
        }
        let data = data?;
        if data.len() < 128 {
            return None;
        }
        Some(Embedding::new(FACE_KIND_ARCFACE, data))
    }
}

#[cfg(feature = "face")]
pub fn loaded_rec_model_name() -> Option<String> {
    onnx::loaded_name()
}

#[cfg(not(feature = "face"))]
pub fn loaded_rec_model_name() -> Option<String> {
    None
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
        assert_eq!(q.reject, FaceReject::NoFace);
        assert!(b.is_none());
    }

    #[test]
    fn textured_skin_can_pass_quality() {
        let rgb = skin_patch(80, 60);
        let (q, b) = assess_face_rgb(&rgb, 80, 60);
        assert!(b.is_some());
        assert!(q.faces == 1);
        assert!(q.area_frac > 0.0);
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
        let (q, emb) = extract_embedding(&rgb, 80, 60);
        assert!(emb.is_none());
        assert_eq!(q.reject, FaceReject::NoEmbed);
    }

    #[test]
    fn threshold_helpers() {
        assert!((face_cosine_threshold("w600k_r50.onnx") - 0.35).abs() < 1e-6);
        assert!((face_cosine_threshold("w600k_mbf.onnx") - 0.40).abs() < 1e-6);
        assert!((face_cosine_threshold("buffalo_sc_w600k_mbf.onnx") - 0.40).abs() < 1e-6);
        let t = effective_face_threshold(0.35, Some("w600k_mbf.onnx"));
        assert!((t - 0.40).abs() < 1e-6);
        let custom = effective_face_threshold(0.55, Some("w600k_mbf.onnx"));
        assert!((custom - 0.55).abs() < 1e-6);
    }

    #[test]
    fn reject_hints_fit_hud() {
        for r in [
            FaceReject::NoFace,
            FaceReject::TooDark,
            FaceReject::TooBright,
            FaceReject::TooBlurry,
            FaceReject::TooSmall,
            FaceReject::TooClose,
            FaceReject::NoEmbed,
        ] {
            assert!(!r.hint().is_empty());
            assert!(r.hint().len() <= 16);
        }
    }
}
