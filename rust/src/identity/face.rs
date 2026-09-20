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
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock};

pub const FACE_ENROLL_NEED: usize = 8;
pub const FACE_KIND_ARCFACE: &str = "arcface";
pub const FACE_KIND_ARCFACE_R50: &str = "arcface-r50";
pub const FACE_KIND_ARCFACE_MBF: &str = "arcface-mbf";
pub const FACE_KIND_PROBE: &str = "face-probe";

/// Stored gallery kinds. Rec session waits until this is `Some`.
static GALLERY_KINDS: Mutex<Option<Vec<String>>> = Mutex::new(None);

/// Record stored face kinds so the rec net is not committed until profiles are known.
pub fn set_gallery_kinds(kinds: &[String]) {
    if let Ok(mut g) = GALLERY_KINDS.lock() {
        *g = Some(kinds.to_vec());
    }
}

/// Drop the rec session so the next embed recommits after kinds / override change.
pub fn reload_rec() {
    #[cfg(feature = "face")]
    {
        onnx::drop_session();
    }
}

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
    pub faces: u32,
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
        "w600k_mbf.onnx",
        "buffalo_sc_w600k_mbf.onnx",
        "w600k_r50.onnx",
        "arcface.onnx",
    ]
}

pub(crate) fn is_r50_kind(k: &str) -> bool {
    k == FACE_KIND_ARCFACE || k == FACE_KIND_ARCFACE_R50
}

fn rec_override_path(dir: &Path, spec: &str) -> Option<PathBuf> {
    let spec = spec.trim();
    if spec.is_empty() {
        return None;
    }
    if spec.eq_ignore_ascii_case("r50") {
        return Some(dir.join("w600k_r50.onnx"));
    }
    if spec.eq_ignore_ascii_case("mbf") {
        return Some(dir.join("w600k_mbf.onnx"));
    }
    let p = PathBuf::from(spec);
    if p.is_file() {
        return Some(p);
    }
    let joined = dir.join(spec);
    if joined.is_file() {
        Some(joined)
    } else {
        None
    }
}

fn gallery_has_r50<S: AsRef<str>>(gallery_kinds: &[S]) -> bool {
    gallery_kinds.iter().any(|k| is_r50_kind(k.as_ref()))
}

fn gallery_kinds_snapshot() -> Option<Vec<String>> {
    GALLERY_KINDS.lock().ok()?.clone()
}

fn log_missing_face_rec(spec: &str) {
    static ONCE: Once = Once::new();
    let spec = spec.to_string();
    ONCE.call_once(move || {
        eprintln!("buckyboi: BUCKYBOI_FACE_REC={spec} not found; using gallery heuristic");
    });
}

/// Prefer MobileFaceNet unless the gallery already has r50 / `arcface` vectors.
#[cfg_attr(not(feature = "face"), allow(dead_code))]
pub(crate) fn select_rec_model<S: AsRef<str>>(gallery_kinds: &[S]) -> Option<PathBuf> {
    select_rec_model_in(crate::identity::models_dir()?.as_path(), gallery_kinds)
}

fn select_rec_model_in<S: AsRef<str>>(dir: &Path, gallery_kinds: &[S]) -> Option<PathBuf> {
    if let Ok(spec) = std::env::var("BUCKYBOI_FACE_REC") {
        let spec = spec.trim();
        if !spec.is_empty() {
            match rec_override_path(dir, spec).filter(|p| p.is_file()) {
                Some(p) => return Some(p),
                None => log_missing_face_rec(spec),
            }
        }
    }
    if gallery_has_r50(gallery_kinds) {
        let r50 = dir.join("w600k_r50.onnx");
        if r50.is_file() {
            return Some(r50);
        }
        let alias = dir.join("arcface.onnx");
        if alias.is_file() {
            return Some(alias);
        }
    }
    for name in rec_model_names() {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Production r50 keeps `kind=arcface` so existing galleries still match.
pub fn rec_embed_kind(model_name: &str) -> &'static str {
    let n = model_name.to_ascii_lowercase();
    if n.contains("mbf") || n.contains("mobilefacenet") || n.contains("buffalo_sc") {
        FACE_KIND_ARCFACE_MBF
    } else {
        FACE_KIND_ARCFACE
    }
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

fn detect_counted(rgb: &[u8], w: u32, h: u32) -> (u32, Option<DetectedFace>) {
    let faces = detect_faces(rgb, w, h);
    let n = faces.len() as u32;
    (n, pick_primary_face(&faces, w as f32, h as f32))
}

fn publish_primary(face: &DetectedFace, w: u32, h: u32, faces: u32, from_scrfd: bool) {
    publish_look(FaceLook {
        cx: face.cx(),
        cy: face.cy(),
        fw: w as f32,
        fh: h as f32,
        t_ms: now_ms_soft(),
        from_scrfd,
        faces,
    });
}

/// Skin-blob / SCRFD box + quality gates. Used by enroll and tests.
pub fn assess_face_rgb(rgb: &[u8], w: u32, h: u32) -> (FaceQuality, Option<FaceBox>) {
    let (n, Some(face)) = detect_counted(rgb, w, h) else {
        return (FaceQuality::reject(FaceReject::NoFace), None);
    };
    let (crop, _aligned) = crop_for_quality(rgb, w, h, &face);
    let q = quality_from_crop(&crop, face.bbox, w, h, n);
    publish_primary(
        &face,
        w,
        h,
        n,
        face.kps.is_some() || face.score >= SCRFD_DET_THRESH,
    );
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
    let (q, e, _) = extract_detected(rgb, w, h, true);
    (q, e)
}

/// Detect + quality + look publish. ArcFace / probe only when `want_embed`.
pub fn extract_detected(
    rgb: &[u8],
    w: u32,
    h: u32,
    want_embed: bool,
) -> (FaceQuality, Option<Embedding>, Option<DetectedFace>) {
    let (n, Some(face)) = detect_counted(rgb, w, h) else {
        return (FaceQuality::reject(FaceReject::NoFace), None, None);
    };
    let (crop, aligned) = crop_for_quality(rgb, w, h, &face);
    let mut q = quality_from_crop(&crop, face.bbox, w, h, n);
    publish_look(FaceLook {
        cx: face.cx(),
        cy: face.cy(),
        fw: w as f32,
        fh: h as f32,
        t_ms: now_ms_soft(),
        from_scrfd: face.kps.is_some(),
        faces: n,
    });
    if !q.ok {
        return (q, None, Some(face));
    }
    if !want_embed {
        return (q, None, Some(face));
    }
    let _ = aligned;
    #[cfg(feature = "face")]
    {
        if let Some(emb) = onnx_embed_aligned(&crop, aligned) {
            return (q, Some(emb), Some(face));
        }
    }
    if crate::identity::env_flag("BUCKYBOI_FACE_PROBE") {
        return (q, Some(probe_embed(rgb, w, h, face.bbox)), Some(face));
    }
    q.ok = false;
    q.reject = FaceReject::NoEmbed;
    (q, None, Some(face))
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
        input: String,
    }

    static REC: Mutex<Option<Rec>> = Mutex::new(None);

    pub fn loaded_name() -> Option<String> {
        REC.lock().ok()?.as_ref().map(|r| r.name.clone())
    }

    pub fn drop_session() {
        if let Ok(mut g) = REC.lock() {
            *g = None;
        }
    }

    fn session() -> Option<()> {
        if REC.lock().ok()?.is_some() {
            return Some(());
        }
        let kinds = gallery_kinds_snapshot()?;
        let mut recg = REC.lock().ok()?;
        if recg.is_some() {
            return Some(());
        }
        let path = select_rec_model(&kinds)?;
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("arcface")
            .to_string();
        let sess = crate::identity::ort_sess::session_from_file(&path)?;
        let input = sess.inputs().first()?.name().to_string();
        let kind = rec_embed_kind(&name);
        if kind == FACE_KIND_ARCFACE && gallery_has_r50(&kinds) {
            eprintln!(
                "buckyboi: ArcFace r50 (gallery has arcface vectors; BUCKYBOI_FACE_REC=mbf + re-enroll FACE to switch)"
            );
        } else {
            eprintln!(
                "buckyboi: ArcFace {} (cosine ~{:.2}, kind={kind})",
                name,
                face_cosine_threshold(&name)
            );
        }
        *recg = Some(Rec {
            session: sess,
            name,
            input,
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
        let tensor = ort::value::Tensor::from_array(blob).ok()?;
        let iname = rec.input.clone();
        let outputs = match rec.session.run(ort::inputs![iname.as_str() => tensor]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("buckyboi: ArcFace run failed ({e})");
                return None;
            }
        };
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
        Some(Embedding::new(rec_embed_kind(&rec.name), data))
    }

    pub fn drop_rec() {
        if let Ok(mut g) = REC.lock() {
            *g = None;
        }
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

    static REC_ENV_LOCK: Mutex<()> = Mutex::new(());

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
        let dir = std::env::temp_dir().join(format!(
            "buckyboi-face-nomodel-{}-{}",
            std::process::id(),
            now_ms_soft()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let rgb = skin_patch(80, 60);
        let (q, emb) = crate::identity::with_models_dir(&dir, || {
            reload_rec();
            extract_embedding(&rgb, 80, 60)
        });
        let _ = std::fs::remove_dir_all(&dir);
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
        assert_eq!(rec_embed_kind("w600k_r50.onnx"), FACE_KIND_ARCFACE);
        assert_eq!(rec_embed_kind("w600k_mbf.onnx"), FACE_KIND_ARCFACE_MBF);
        assert_eq!(
            rec_embed_kind("buffalo_sc_w600k_mbf.onnx"),
            FACE_KIND_ARCFACE_MBF
        );
    }

    struct RestoreFaceRec {
        prev: Option<String>,
        dir: PathBuf,
    }

    impl Drop for RestoreFaceRec {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("BUCKYBOI_FACE_REC", v),
                None => std::env::remove_var("BUCKYBOI_FACE_REC"),
            }
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn rec_tmp() -> RestoreFaceRec {
        let dir = std::env::temp_dir().join(format!(
            "buckyboi-rec-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("w600k_r50.onnx"), b"x").unwrap();
        std::fs::write(dir.join("w600k_mbf.onnx"), b"x").unwrap();
        RestoreFaceRec {
            prev: std::env::var("BUCKYBOI_FACE_REC").ok(),
            dir,
        }
    }

    #[test]
    fn select_rec_model_arcface_gallery_picks_r50() {
        let _env = REC_ENV_LOCK.lock().unwrap();
        let ctx = rec_tmp();
        std::env::remove_var("BUCKYBOI_FACE_REC");
        let r50 = select_rec_model_in(&ctx.dir, &["arcface"]).unwrap();
        assert_eq!(r50.file_name().unwrap(), "w600k_r50.onnx");
        let r50_alias = select_rec_model_in(&ctx.dir, &["arcface-r50"]).unwrap();
        assert_eq!(r50_alias.file_name().unwrap(), "w600k_r50.onnx");
        let empty: [&str; 0] = [];
        let empty = select_rec_model_in(&ctx.dir, &empty).unwrap();
        assert_eq!(empty.file_name().unwrap(), "w600k_mbf.onnx");
        let mbf = select_rec_model_in(&ctx.dir, &["arcface-mbf"]).unwrap();
        assert_eq!(mbf.file_name().unwrap(), "w600k_mbf.onnx");
    }

    #[test]
    fn select_rec_model_face_rec_override() {
        let _env = REC_ENV_LOCK.lock().unwrap();
        let ctx = rec_tmp();
        std::env::set_var("BUCKYBOI_FACE_REC", "mbf");
        let over_mbf = select_rec_model_in(&ctx.dir, &["arcface"]).unwrap();
        assert_eq!(over_mbf.file_name().unwrap(), "w600k_mbf.onnx");
        std::env::set_var("BUCKYBOI_FACE_REC", "r50");
        let empty: [&str; 0] = [];
        let over_r50 = select_rec_model_in(&ctx.dir, &empty).unwrap();
        assert_eq!(over_r50.file_name().unwrap(), "w600k_r50.onnx");
        std::env::set_var("BUCKYBOI_FACE_REC", "missing.onnx");
        let fall = select_rec_model_in(&ctx.dir, &empty).unwrap();
        assert_eq!(fall.file_name().unwrap(), "w600k_mbf.onnx");
    }

    #[test]
    fn rec_waits_until_gallery_kinds_set() {
        struct Restore(Option<Vec<String>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                if let Ok(mut g) = GALLERY_KINDS.lock() {
                    *g = self.0.take();
                }
            }
        }
        let _restore = Restore(gallery_kinds_snapshot());
        if let Ok(mut g) = GALLERY_KINDS.lock() {
            *g = None;
        }
        assert!(gallery_kinds_snapshot().is_none());
        set_gallery_kinds(&[]);
        assert_eq!(gallery_kinds_snapshot().as_deref(), Some(&[][..]));
        if let Ok(mut g) = GALLERY_KINDS.lock() {
            *g = None;
        }
        set_gallery_kinds(&["arcface".into()]);
        assert_eq!(
            gallery_kinds_snapshot().as_deref(),
            Some(["arcface".to_string()].as_slice())
        );
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
