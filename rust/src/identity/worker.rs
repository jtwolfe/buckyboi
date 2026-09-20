//! Vision worker: ONNX off the present thread. FaceSnap + HandSnap only (PR1).

use crate::identity::embed::Embedding;
use crate::identity::enroll::EnrollKind;
use crate::identity::face::{FaceLook, FaceQuality};
use crate::identity::hands::HandStatus;
#[cfg(any(test, feature = "face", feature = "hands", feature = "voice"))]
use crate::identity::VISION_INFER_MS;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
#[cfg(any(feature = "face", feature = "hands", feature = "voice"))]
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "face")]
const LIVE_REC_MS: u64 = 160;
#[cfg_attr(not(feature = "face"), allow(dead_code))]
const SNAP_FRESH_MS: u64 = 400;

#[derive(Clone, Debug)]
pub struct FaceSnap {
    pub t_ms: u64,
    pub quality: FaceQuality,
    pub embedding: Option<Embedding>,
    pub look: Option<FaceLook>,
}

#[derive(Clone, Debug)]
pub struct HandSnap {
    pub t_ms: u64,
    pub status: HandStatus,
}

static FACE_SLOT: OnceLock<Mutex<Option<FaceSnap>>> = OnceLock::new();
static HAND_SLOT: OnceLock<Mutex<Option<HandSnap>>> = OnceLock::new();
static GALLERY_KINDS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
static HANDLE: OnceLock<Mutex<Option<JoinHandle<()>>>> = OnceLock::new();

#[cfg_attr(
    not(any(feature = "face", feature = "hands", feature = "voice")),
    allow(dead_code)
)]
static WORKER_STOP: AtomicBool = AtomicBool::new(false);
#[cfg_attr(
    not(any(feature = "face", feature = "hands", feature = "voice")),
    allow(dead_code)
)]
static VISION_WORKER: AtomicBool = AtomicBool::new(false);
static DEAD_LOGGED: AtomicBool = AtomicBool::new(false);
#[cfg_attr(not(feature = "face"), allow(dead_code))]
static ENROLLING_FACE: AtomicBool = AtomicBool::new(false);
#[allow(dead_code)] // PR2 voice cadence
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

/// Stamp at slice start (miss or hit). Never rewrite after infer.
pub fn stamp_if_due(last_ms: &mut u64, now: u64, period_ms: u64) -> bool {
    if now.saturating_sub(*last_ms) >= period_ms {
        *last_ms = now;
        true
    } else {
        false
    }
}

pub fn latest_face() -> Option<FaceSnap> {
    face_slot().lock().ok().and_then(|g| g.clone())
}

pub fn latest_hand() -> Option<HandSnap> {
    hand_slot().lock().ok().and_then(|g| g.clone())
}

/// Fresh SCRFD/skin look within 400 ms. No GazeSnap in PR1.
pub fn publishing_gaze() -> bool {
    #[cfg(feature = "face")]
    {
        crate::identity::face::latest_look(now_ms(), SNAP_FRESH_MS).is_some()
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
    WORKER_STOP.store(false, Ordering::SeqCst);
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

#[cfg(any(feature = "face", feature = "hands", feature = "voice"))]
fn vision_loop() {
    let mut last_vis_ms = 0u64;
    #[cfg(feature = "face")]
    let mut last_rec_ms = 0u64;
    let mut last_pub_ms = 0u64;
    let mut last_hb_log = 0u64;
    #[cfg(feature = "hands")]
    let mut last_nohand_log = 0u64;
    #[cfg(feature = "hands")]
    let mut nohand_since: Option<u64> = None;

    loop {
        if WORKER_STOP.load(Ordering::SeqCst) {
            break;
        }
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

        let frame = crate::camera::latest_frame();

        if stamp_if_due(&mut last_vis_ms, now, VISION_INFER_MS) {
            #[cfg(feature = "face")]
            if let Some(ref frame) = frame {
                let enrolling_face = ENROLLING_FACE.load(Ordering::Relaxed);
                let rec_due = enrolling_face || now.saturating_sub(last_rec_ms) >= LIVE_REC_MS;
                if rec_due {
                    last_rec_ms = now;
                }
                let (quality, embedding, look) =
                    crate::identity::face::extract_parts(&frame.rgb, frame.w, frame.h, rec_due);
                publish_face(FaceSnap {
                    t_ms: now,
                    quality,
                    embedding,
                    look,
                });
            }

            #[cfg(feature = "hands")]
            {
                let _last_hand_infer_ms = now; // stamp before extract, miss or hit
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
                let _ = _last_hand_infer_ms;
            }
        }

        last_pub_ms = now_ms();
        let spent = t0.elapsed().as_millis() as u64;
        let rest = VISION_INFER_MS.saturating_sub(spent);
        if rest > 0 {
            std::thread::sleep(Duration::from_millis(rest));
        }
    }
    VISION_WORKER.store(false, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::face::FaceReject;

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
            look: None,
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
}
