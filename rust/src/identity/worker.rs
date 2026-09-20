//! Vision worker: ONNX / sherpa off the present thread. Face / hand / gaze / voice snaps.

#[cfg(feature = "face")]
use crate::gaze::face_to_screen;
use crate::identity::embed::Embedding;
use crate::identity::enroll::EnrollKind;
use crate::identity::face::FaceQuality;
use crate::identity::gaze_calib::GazeCalib;
use crate::identity::hands::HandStatus;
use crate::identity::voice::VoiceQuality;
use crate::identity::VISION_INFER_MS;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
#[cfg(any(feature = "face", feature = "hands", feature = "voice"))]
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "face")]
const LIVE_REC_MS: u64 = 160;
#[cfg(any(feature = "face", feature = "hands", feature = "voice"))]
const FRAME_FRESH_MS: u64 = 500;

#[derive(Clone, Debug)]
pub struct FaceSnap {
    pub t_ms: u64,
    pub quality: FaceQuality,
    pub embedding: Option<Embedding>,
}

#[derive(Clone, Debug)]
pub struct HandSnap {
    pub t_ms: u64,
    pub status: HandStatus,
}

#[derive(Clone, Debug)]
pub struct VoiceSnap {
    pub t_ms: u64,
    pub quality: VoiceQuality,
    pub embedding: Option<Embedding>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GazeKind {
    MeshCalibrated,
    MeshUncalibrated,
    FaceBox,
}

#[derive(Clone, Debug)]
pub struct GazeSnap {
    pub t_ms: u64,
    pub sx: f32,
    pub sy: f32,
    pub kind: GazeKind,
    pub iris_nx: f32,
    pub iris_ny: f32,
    pub nose_nx: f32,
    pub nose_ny: f32,
    pub iris_l: bool,
    pub iris_r: bool,
    pub ok: bool,
}

static FACE_SLOT: OnceLock<Mutex<Option<FaceSnap>>> = OnceLock::new();
static HAND_SLOT: OnceLock<Mutex<Option<HandSnap>>> = OnceLock::new();
static VOICE_SLOT: OnceLock<Mutex<Option<VoiceSnap>>> = OnceLock::new();
static GAZE_SLOT: OnceLock<Mutex<Option<GazeSnap>>> = OnceLock::new();
static CALIB_SLOT: OnceLock<Mutex<Option<GazeCalib>>> = OnceLock::new();
static GALLERY_KINDS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
static HANDLE: OnceLock<Mutex<Option<JoinHandle<()>>>> = OnceLock::new();
#[cfg_attr(not(feature = "face"), allow(dead_code))]
static GAZE_LOGGED: Mutex<u8> = Mutex::new(0);
#[cfg_attr(not(feature = "face"), allow(dead_code))]
static CAM_MISMATCH_LOGGED: AtomicBool = AtomicBool::new(false);

#[cfg_attr(
    not(any(feature = "face", feature = "hands", feature = "voice")),
    allow(dead_code)
)]
static VISION_WORKER: AtomicBool = AtomicBool::new(false);
static DEAD_LOGGED: AtomicBool = AtomicBool::new(false);
#[cfg_attr(not(feature = "face"), allow(dead_code))]
static ENROLLING_FACE: AtomicBool = AtomicBool::new(false);
#[cfg_attr(not(feature = "voice"), allow(dead_code))]
static ENROLLING_VOICE: AtomicBool = AtomicBool::new(false);
static SCREEN_W: AtomicU32 = AtomicU32::new(0);
static SCREEN_H: AtomicU32 = AtomicU32::new(0);
static WATCH_LAST_MS: AtomicU64 = AtomicU64::new(0);

#[cfg_attr(not(feature = "face"), allow(dead_code))]
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn face_slot() -> &'static Mutex<Option<FaceSnap>> {
    FACE_SLOT.get_or_init(|| Mutex::new(None))
}

fn hand_slot() -> &'static Mutex<Option<HandSnap>> {
    HAND_SLOT.get_or_init(|| Mutex::new(None))
}

fn voice_slot() -> &'static Mutex<Option<VoiceSnap>> {
    VOICE_SLOT.get_or_init(|| Mutex::new(None))
}

fn gaze_slot() -> &'static Mutex<Option<GazeSnap>> {
    GAZE_SLOT.get_or_init(|| Mutex::new(None))
}

fn calib_slot() -> &'static Mutex<Option<GazeCalib>> {
    CALIB_SLOT.get_or_init(|| Mutex::new(GazeCalib::load()))
}

fn kinds_slot() -> &'static Mutex<Vec<String>> {
    GALLERY_KINDS.get_or_init(|| Mutex::new(Vec::new()))
}

fn handle_slot() -> &'static Mutex<Option<JoinHandle<()>>> {
    HANDLE.get_or_init(|| Mutex::new(None))
}

#[cfg_attr(not(feature = "face"), allow(dead_code))]
fn publish_face(snap: FaceSnap) {
    if let Ok(mut g) = face_slot().lock() {
        *g = Some(snap);
    }
}

#[cfg_attr(not(feature = "hands"), allow(dead_code))]
fn publish_hand(snap: HandSnap) {
    if let Ok(mut g) = hand_slot().lock() {
        *g = Some(snap);
    }
}

#[cfg_attr(not(feature = "voice"), allow(dead_code))]
fn publish_voice(snap: VoiceSnap) {
    if let Ok(mut g) = voice_slot().lock() {
        *g = Some(snap);
    }
}

#[cfg_attr(not(any(test, feature = "face")), allow(dead_code))]
fn publish_gaze(snap: GazeSnap) {
    if let Ok(mut g) = gaze_slot().lock() {
        *g = Some(snap);
    }
}

/// Stamp at slice start (miss or hit). Never rewrite after infer.
#[cfg_attr(
    not(any(feature = "face", feature = "hands", feature = "voice")),
    allow(dead_code)
)]
fn stamp_if_due(last_ms: &mut u64, now: u64, period_ms: u64) -> bool {
    if now.saturating_sub(*last_ms) >= period_ms {
        *last_ms = now;
        true
    } else {
        false
    }
}

/// Consume a one-shot vis skip (voice was deferred). Does not stamp `last_vis_ms`.
#[cfg_attr(
    not(any(feature = "face", feature = "hands", feature = "voice")),
    allow(dead_code)
)]
fn should_run_vis(skip_vis: &mut bool, last_vis_ms: &mut u64, now: u64, period_ms: u64) -> bool {
    if *skip_vis {
        *skip_vis = false;
        return false;
    }
    stamp_if_due(last_vis_ms, now, period_ms)
}

pub fn latest_face() -> Option<FaceSnap> {
    face_slot().lock().ok().and_then(|g| g.clone())
}

pub fn latest_hand() -> Option<HandSnap> {
    hand_slot().lock().ok().and_then(|g| g.clone())
}

pub fn latest_voice() -> Option<VoiceSnap> {
    voice_slot().lock().ok().and_then(|g| g.clone())
}

/// Clone the latest gaze snap if it landed within `max_age_ms`.
pub fn latest_gaze(now_ms: u64, max_age_ms: u64) -> Option<GazeSnap> {
    let g = gaze_slot().lock().ok()?;
    let snap = (*g).clone()?;
    if now_ms.saturating_sub(snap.t_ms) <= max_age_ms {
        Some(snap)
    } else {
        None
    }
}

pub fn clear_gaze() {
    if let Ok(mut g) = gaze_slot().lock() {
        *g = None;
    }
}

pub fn set_calib(calib: Option<GazeCalib>) {
    CAM_MISMATCH_LOGGED.store(false, Ordering::Relaxed);
    if let Ok(mut g) = calib_slot().lock() {
        *g = calib;
    }
}

#[cfg_attr(not(any(test, feature = "face")), allow(dead_code))]
fn current_calib() -> Option<GazeCalib> {
    calib_slot().lock().ok().and_then(|g| g.clone())
}

/// Fresh mesh/box gaze or SCRFD look within 400 ms.
pub fn publishing_gaze() -> bool {
    #[cfg(feature = "face")]
    {
        let now = now_ms();
        latest_gaze(now, 400).is_some() || crate::identity::face::latest_look(now, 400).is_some()
    }
    #[cfg(not(feature = "face"))]
    {
        false
    }
}

pub fn set_screen(w: u32, h: u32) {
    SCREEN_W.store(w, Ordering::Relaxed);
    SCREEN_H.store(h, Ordering::Relaxed);
}

pub fn screen_size() -> (u32, u32) {
    (
        SCREEN_W.load(Ordering::Relaxed),
        SCREEN_H.load(Ordering::Relaxed),
    )
}

pub fn set_gallery_kinds(kinds: &[String]) {
    crate::identity::face::set_gallery_kinds(kinds);
    if let Ok(mut g) = kinds_slot().lock() {
        *g = kinds.to_vec();
    }
}

pub fn gallery_kinds() -> Vec<String> {
    kinds_slot().lock().map(|g| g.clone()).unwrap_or_default()
}

pub fn reload_rec() {
    crate::identity::face::reload_rec();
}

pub fn set_enrolling(kind: EnrollKind, on: bool) {
    match kind {
        EnrollKind::Face => ENROLLING_FACE.store(on, Ordering::Relaxed),
        EnrollKind::Voice => ENROLLING_VOICE.store(on, Ordering::Relaxed),
        EnrollKind::Gesture => {}
    }
}

#[cfg_attr(
    not(any(feature = "face", feature = "hands", feature = "voice")),
    allow(dead_code)
)]
fn ort_intra() -> usize {
    #[cfg(any(feature = "face", feature = "hands"))]
    {
        crate::identity::ort_sess::intra_threads()
    }
    #[cfg(not(any(feature = "face", feature = "hands")))]
    {
        1
    }
}

/// No-op unless `face`, `hands`, or `voice` is compiled in.
pub fn start_vision_worker(gallery_kinds: &[String]) {
    set_gallery_kinds(gallery_kinds);
    #[cfg(any(feature = "face", feature = "hands", feature = "voice"))]
    spawn_if_needed();
}

#[cfg(any(feature = "face", feature = "hands", feature = "voice"))]
fn spawn_if_needed() {
    if VISION_WORKER.load(Ordering::SeqCst) {
        if let Ok(g) = handle_slot().lock() {
            if let Some(h) = g.as_ref() {
                if !h.is_finished() {
                    return;
                }
            }
        }
    }
    DEAD_LOGGED.store(false, Ordering::SeqCst);
    VISION_WORKER.store(true, Ordering::SeqCst);
    let intra = ort_intra();
    match std::thread::Builder::new()
        .name("buckyboi-vision".into())
        .spawn(vision_loop)
    {
        Ok(h) => {
            eprintln!("buckyboi: vision worker on (ort intra={intra})");
            if let Ok(mut g) = handle_slot().lock() {
                *g = Some(h);
            }
        }
        Err(e) => {
            VISION_WORKER.store(false, Ordering::SeqCst);
            eprintln!("buckyboi: could not spawn vision worker ({e})");
        }
    }
}

pub fn watch_vision_worker(now: u64) {
    if now.saturating_sub(WATCH_LAST_MS.load(Ordering::Relaxed)) < 1_000 {
        return;
    }
    WATCH_LAST_MS.store(now, Ordering::Relaxed);
    let Ok(g) = handle_slot().lock() else {
        return;
    };
    let Some(h) = g.as_ref() else {
        return;
    };
    if h.is_finished() {
        if !DEAD_LOGGED.swap(true, Ordering::SeqCst) {
            eprintln!("buckyboi: vision worker died");
        }
        VISION_WORKER.store(false, Ordering::SeqCst);
        clear_gaze();
    }
}

#[cfg(feature = "hands")]
fn hand_sim_class() -> Option<crate::identity::hands::GestureClass> {
    let sim = std::env::var("BUCKYBOI_HAND_SIM").ok()?;
    let cls = crate::identity::hands::GestureClass::parse(&sim);
    if cls == crate::identity::hands::GestureClass::Unknown {
        None
    } else {
        Some(cls)
    }
}

#[cfg(feature = "hands")]
fn hand_status(frame: Option<&crate::camera::CamFrame>) -> HandStatus {
    if let Some(cls) = hand_sim_class() {
        return HandStatus::Ok(crate::identity::hands::synthetic(cls));
    }
    let Some(frame) = frame else {
        return if crate::identity::hands::models_present() {
            HandStatus::NoHand
        } else {
            HandStatus::NoModel
        };
    };
    crate::identity::hands::extract_status(&frame.rgb, frame.w, frame.h)
}

/// Remainder of the vis cadence, or 0 when voice is deferred so the follow-up loop
/// (which also skips vis via `skip_vis`) can embed without waiting out 80 ms.
#[cfg_attr(
    not(any(feature = "face", feature = "hands", feature = "voice")),
    allow(dead_code)
)]
fn vis_sleep_ms(spent_ms: u64, defer_voice: bool) -> u64 {
    if defer_voice {
        0
    } else {
        VISION_INFER_MS.saturating_sub(spent_ms)
    }
}

#[cfg(feature = "face")]
fn log_gaze_kind(kind: GazeKind, extra: &str) {
    let tag = match kind {
        GazeKind::MeshCalibrated => 1u8,
        GazeKind::MeshUncalibrated => 2,
        GazeKind::FaceBox => 3,
    };
    if let Ok(mut g) = GAZE_LOGGED.lock() {
        if *g != tag {
            *g = tag;
            match kind {
                GazeKind::MeshCalibrated => {
                    eprintln!("buckyboi: gaze: landmarker+calib {extra}");
                }
                GazeKind::MeshUncalibrated => {
                    eprintln!("buckyboi: gaze: uncalibrated {extra}");
                }
                GazeKind::FaceBox => {
                    eprintln!("buckyboi: gaze: face-box fallback {extra}");
                }
            }
        }
    }
}

#[cfg(feature = "face")]
fn map_gaze(
    now: u64,
    frame: &crate::camera::CamFrame,
    face: &crate::identity::scrfd::DetectedFace,
) -> GazeSnap {
    let (sw, sh) = screen_size();
    let sw = sw as f32;
    let sh = sh as f32;
    let fw = frame.w as f32;
    let fh = frame.h as f32;
    let feat = crate::identity::gaze_track::infer(&frame.rgb, frame.w, frame.h, face);
    let mut snap = GazeSnap {
        t_ms: now,
        sx: 0.0,
        sy: 0.0,
        kind: GazeKind::FaceBox,
        iris_nx: feat.map(|f| f.iris_nx).unwrap_or(0.5),
        iris_ny: feat.map(|f| f.iris_ny).unwrap_or(0.5),
        nose_nx: feat.map(|f| f.nose_nx).unwrap_or(0.5),
        nose_ny: feat.map(|f| f.nose_ny).unwrap_or(0.5),
        iris_l: feat.map(|f| f.iris_l).unwrap_or(false),
        iris_r: feat.map(|f| f.iris_r).unwrap_or(false),
        ok: feat.map(|f| f.ok).unwrap_or(false),
    };
    if let Some(f) = feat.filter(|f| f.ok) {
        snap.iris_nx = f.iris_nx;
        snap.iris_ny = f.iris_ny;
        snap.nose_nx = f.nose_nx;
        snap.nose_ny = f.nose_ny;
        snap.iris_l = f.iris_l;
        snap.iris_r = f.iris_r;
        snap.ok = true;
        if sw > 1.0 && sh > 1.0 {
            if let Some(c) = current_calib() {
                if c.is_stale(sw as u32, sh as u32) {
                    log_gaze_kind(GazeKind::MeshUncalibrated, "calib stale (screen)");
                    let (sx, sy) = crate::gaze::uncalibrated(
                        f.iris_nx, f.iris_ny, f.nose_nx, f.nose_ny, sw, sh,
                    );
                    snap.sx = sx;
                    snap.sy = sy;
                    snap.kind = GazeKind::MeshUncalibrated;
                    return snap;
                }
                if c.camera != crate::identity::gaze_calib::camera_label()
                    && !CAM_MISMATCH_LOGGED.swap(true, Ordering::Relaxed)
                {
                    eprintln!("buckyboi: gaze: camera path mismatch");
                }
                if let Some((sx, sy)) = c.apply(f.iris_nx, f.iris_ny, f.nose_nx, f.nose_ny) {
                    log_gaze_kind(
                        GazeKind::MeshCalibrated,
                        &format!("rmse={:.0}px", c.rmse_px),
                    );
                    snap.sx = sx.clamp(0.0, sw.max(1.0));
                    snap.sy = sy.clamp(0.0, sh.max(1.0));
                    snap.kind = GazeKind::MeshCalibrated;
                    return snap;
                }
            }
            log_gaze_kind(GazeKind::MeshUncalibrated, "");
            let (sx, sy) =
                crate::gaze::uncalibrated(f.iris_nx, f.iris_ny, f.nose_nx, f.nose_ny, sw, sh);
            snap.sx = sx;
            snap.sy = sy;
            snap.kind = GazeKind::MeshUncalibrated;
            return snap;
        }
    }
    log_gaze_kind(GazeKind::FaceBox, "");
    let (sx, sy) = if sw > 1.0 && sh > 1.0 {
        face_to_screen(face.cx(), face.cy(), fw, fh, sw, sh, true)
    } else {
        (0.0, 0.0)
    };
    snap.sx = sx;
    snap.sy = sy;
    snap.kind = GazeKind::FaceBox;
    snap
}

#[cfg(any(feature = "face", feature = "hands", feature = "voice"))]
fn vision_loop() {
    let mut last_vis_ms = 0u64;
    #[cfg(feature = "face")]
    let mut last_rec_ms = 0u64;
    #[cfg(feature = "voice")]
    let mut last_voice_ms = 0u64;
    let mut last_pub_ms = 0u64;
    let mut last_hb_log = 0u64;
    #[cfg(feature = "hands")]
    let mut last_nohand_log = 0u64;
    #[cfg(feature = "hands")]
    let mut nohand_since: Option<u64> = None;
    // After deferring voice, skip the next vis stamp so embed can run even if vis ≥ 80 ms.
    let mut skip_vis = false;

    loop {
        let t0 = Instant::now();
        let now = now_ms();

        if last_pub_ms > 0
            && now.saturating_sub(last_pub_ms) > 500
            && now.saturating_sub(last_hb_log) >= 5_000
        {
            eprintln!(
                "buckyboi: vision heartbeat stale {} ms",
                now.saturating_sub(last_pub_ms)
            );
            last_hb_log = now;
        }

        let frame =
            crate::camera::latest_frame().filter(|f| now.saturating_sub(f.t_ms) <= FRAME_FRESH_MS);

        if should_run_vis(&mut skip_vis, &mut last_vis_ms, now, VISION_INFER_MS) {
            #[cfg(feature = "face")]
            if let Some(ref frame) = frame {
                let enrolling_face = ENROLLING_FACE.load(Ordering::Relaxed);
                let rec_due = enrolling_face || now.saturating_sub(last_rec_ms) >= LIVE_REC_MS;
                if rec_due {
                    last_rec_ms = now;
                }
                let (quality, embedding, face) =
                    crate::identity::face::extract_detected(&frame.rgb, frame.w, frame.h, rec_due);
                publish_face(FaceSnap {
                    t_ms: now,
                    quality,
                    embedding,
                });
                if let Some(face) = face {
                    publish_gaze(map_gaze(now, frame, &face));
                }
            }

            #[cfg(feature = "hands")]
            {
                let status = hand_status(frame.as_ref());
                match &status {
                    HandStatus::NoHand => {
                        if nohand_since.is_none() {
                            nohand_since = Some(now);
                        }
                        if let Some(since) = nohand_since {
                            if now.saturating_sub(since) > 1_000
                                && now.saturating_sub(last_nohand_log) >= 5_000
                            {
                                eprintln!("buckyboi: hands: no palm (throttled)");
                                last_nohand_log = now;
                            }
                        }
                    }
                    _ => nohand_since = None,
                }
                publish_hand(HandSnap { t_ms: now, status });
            }
        }

        // Voice cadence is separate. If due but this slice already spent ≥ 40 ms,
        // do not stamp last_voice_ms and sleep 0 so the next loop skips vision.
        #[cfg_attr(not(feature = "voice"), allow(unused_mut))]
        let mut defer_voice = false;
        #[cfg(feature = "voice")]
        {
            let period = if ENROLLING_VOICE.load(Ordering::Relaxed) {
                1_600
            } else {
                2_400
            };
            if now.saturating_sub(last_voice_ms) >= period {
                if t0.elapsed().as_millis() < 40 {
                    last_voice_ms = now;
                    let samples = crate::identity::voice::worker_samples();
                    let (quality, embedding) = crate::identity::voice::extract_embedding(
                        &samples,
                        crate::identity::voice::VOICE_SAMPLE_RATE,
                    );
                    publish_voice(VoiceSnap {
                        t_ms: now,
                        quality,
                        embedding,
                    });
                } else {
                    defer_voice = true;
                    skip_vis = true;
                }
            }
        }

        last_pub_ms = now_ms();
        let spent = t0.elapsed().as_millis() as u64;
        let rest = vis_sleep_ms(spent, defer_voice);
        if rest > 0 {
            std::thread::sleep(Duration::from_millis(rest));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::face::FaceReject;

    #[test]
    fn vis_sleep_zero_when_voice_deferred() {
        assert_eq!(vis_sleep_ms(50, true), 0);
        assert_eq!(vis_sleep_ms(50, false), VISION_INFER_MS - 50);
        assert_eq!(vis_sleep_ms(90, false), 0);
        assert_eq!(vis_sleep_ms(0, true), 0);
    }

    #[test]
    fn skip_vis_one_shot_does_not_stamp_even_when_due() {
        let mut skip = false;
        let mut last = 0u64;
        assert!(should_run_vis(
            &mut skip,
            &mut last,
            VISION_INFER_MS,
            VISION_INFER_MS
        ));
        assert_eq!(last, VISION_INFER_MS);

        skip = true;
        // Overrun: next loop is immediately due, but skip wins and must not stamp.
        assert!(!should_run_vis(
            &mut skip,
            &mut last,
            VISION_INFER_MS * 2,
            VISION_INFER_MS
        ));
        assert_eq!(last, VISION_INFER_MS);
        assert!(!skip);

        assert!(should_run_vis(
            &mut skip,
            &mut last,
            VISION_INFER_MS * 2,
            VISION_INFER_MS
        ));
        assert_eq!(last, VISION_INFER_MS * 2);
    }

    #[test]
    fn stamp_if_due_stamps_before_miss() {
        let mut last = 0u64;
        assert!(stamp_if_due(&mut last, VISION_INFER_MS, VISION_INFER_MS));
        assert_eq!(last, VISION_INFER_MS);
        // Infer missed; stamp already advanced so the next 16 ms tick is skipped.
        assert!(!stamp_if_due(
            &mut last,
            VISION_INFER_MS + 16,
            VISION_INFER_MS
        ));
        assert_eq!(last, VISION_INFER_MS);
        assert!(stamp_if_due(
            &mut last,
            VISION_INFER_MS * 2,
            VISION_INFER_MS
        ));
        assert_eq!(last, VISION_INFER_MS * 2);
    }

    #[test]
    fn latest_face_clones_out_of_lock() {
        let snap = FaceSnap {
            t_ms: 1234,
            quality: FaceQuality::reject(FaceReject::NoFace),
            embedding: None,
        };
        publish_face(snap);
        let got = latest_face().expect("published");
        assert_eq!(got.t_ms, 1234);
        assert_eq!(got.quality.reject, FaceReject::NoFace);
        assert!(got.embedding.is_none());
    }

    #[test]
    fn latest_hand_visible_after_publish() {
        publish_hand(HandSnap {
            t_ms: 99,
            status: HandStatus::NoHand,
        });
        let got = latest_hand().expect("published");
        assert_eq!(got.t_ms, 99);
        assert!(matches!(got.status, HandStatus::NoHand));
        publish_hand(HandSnap {
            t_ms: 100,
            status: HandStatus::NoModel,
        });
        let got = latest_hand().expect("published");
        assert!(matches!(got.status, HandStatus::NoModel));
    }

    #[test]
    fn latest_voice_clones_out_of_lock() {
        let snap = VoiceSnap {
            t_ms: 42,
            quality: VoiceQuality::assess(&[], crate::identity::voice::VOICE_SAMPLE_RATE),
            embedding: None,
        };
        publish_voice(snap);
        let got = latest_voice().expect("published");
        assert_eq!(got.t_ms, 42);
        assert!(!got.quality.ok);
        assert!(got.embedding.is_none());
        publish_voice(VoiceSnap {
            t_ms: 43,
            quality: VoiceQuality::assess(&[], crate::identity::voice::VOICE_SAMPLE_RATE),
            embedding: Some(Embedding::new("logmel", vec![1.0; 4])),
        });
        let got = latest_voice().expect("published");
        assert_eq!(got.t_ms, 43);
        assert!(got.embedding.is_some());
    }

    #[test]
    fn set_screen_and_kinds_roundtrip() {
        set_screen(1920, 1054);
        assert_eq!(screen_size(), (1920, 1054));
        set_gallery_kinds(&["arcface".into()]);
        assert_eq!(gallery_kinds(), vec!["arcface".to_string()]);
        set_enrolling(EnrollKind::Face, true);
        set_enrolling(EnrollKind::Face, false);
        set_enrolling(EnrollKind::Voice, true);
        set_enrolling(EnrollKind::Voice, false);
    }

    fn dummy_gaze(t_ms: u64) -> GazeSnap {
        GazeSnap {
            t_ms,
            sx: 100.0,
            sy: 200.0,
            kind: GazeKind::FaceBox,
            iris_nx: 0.4,
            iris_ny: 0.5,
            nose_nx: 0.5,
            nose_ny: 0.5,
            iris_l: true,
            iris_r: true,
            ok: true,
        }
    }

    #[test]
    fn latest_gaze_respects_max_age_and_clear() {
        publish_gaze(dummy_gaze(1_000));
        let got = latest_gaze(1_000, 400).expect("fresh");
        assert_eq!(got.t_ms, 1_000);
        assert!((got.sx - 100.0).abs() < 1e-3);
        assert!(latest_gaze(1_400, 400).is_some());
        assert!(latest_gaze(1_401, 400).is_none());
        clear_gaze();
        assert!(latest_gaze(1_000, 400).is_none());
        assert!(latest_gaze(1_401, 400).is_none());
    }

    #[test]
    fn set_calib_roundtrip() {
        let c = GazeCalib {
            version: 1,
            screen_w: 1920,
            screen_h: 1054,
            camera: "/dev/video0".into(),
            model: "face_landmarker".into(),
            points: 5,
            affine: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            ridge_w: None,
            rmse_px: 12.0,
            created_ms: 0,
        };
        set_calib(Some(c.clone()));
        let got = current_calib().expect("set");
        assert_eq!(got.rmse_px, 12.0);
        set_calib(None);
        assert!(current_calib().is_none());
    }
}
