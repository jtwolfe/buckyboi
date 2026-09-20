//! Local multi-person identity: face, voice, and gesture.
//!
//! Matching is fail-closed. Embeddings live under `~/.config/buckyboi/`.
//! ONNX / microphone backends are feature-gated; the state machines always compile.

pub mod align;
pub mod embed;
pub mod enroll;
pub mod face;
pub mod gate;
pub mod hands;
#[cfg(any(feature = "face", feature = "hands"))]
pub(crate) mod ort_sess;
pub mod palm;
pub mod persist;
pub mod scrfd;
pub mod voice;
pub mod wizard;
pub mod worker;

pub use embed::*;
pub use enroll::*;
pub use face::{
    assess_face_rgb, extract_embedding as extract_face, latest_look, reload_rec, set_gallery_kinds,
    FaceQuality, FaceReject, FACE_ENROLL_NEED, FACE_KIND_ARCFACE, FACE_KIND_ARCFACE_MBF,
    FACE_KIND_ARCFACE_R50, FACE_KIND_PROBE, FACE_THRESHOLD_MBF, FACE_THRESHOLD_R50,
};
pub use gate::*;
pub use hands::{
    classify_gesture, classify_rules, classify_trained, extract_status, gesture_centroid,
    models_present, GestureAction, GestureClass, GestureMap, GestureSample, HandLandmarks,
    HandStatus, DEFAULT_GESTURE_MAP,
};
pub use persist::*;
pub use voice::{
    input_available, logmel_embed, pick_speaker_model, speech_energy, start_mic,
    voice_cosine_threshold, VoiceQuality, VoiceReject, VOICE_ENROLL_NEED, VOICE_ENROLL_REJECT_CAP,
    VOICE_ENROLL_TIMEOUT_MS, VOICE_MIN_MS, VOICE_THRESHOLD_SHERPA,
};
pub use wizard::{skip_on_enroll_fail, FirstRunWizard, WizardEvent, WizardStep, LARGEST_FACE_CHIP};
pub use worker::{
    latest_face, latest_hand, latest_voice, publishing_gaze, reload_rec, set_enrolling,
    set_gallery_kinds, set_screen, start_vision_worker, watch_vision_worker, FaceSnap, HandSnap,
    VoiceSnap,
};

/// Vision / identity ONNX cadence (~12.5 Hz) so the overlay stays at 60 Hz.
pub const VISION_INFER_MS: u64 = 80;

use std::path::PathBuf;

/// Config root. Override with `BUCKYBOI_CONFIG` (tests).
pub fn config_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BUCKYBOI_CONFIG") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".config").join("buckyboi"))
}

pub fn models_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BUCKYBOI_MODELS") {
        return Some(PathBuf::from(p));
    }
    Some(config_dir()?.join("models"))
}

pub fn env_or_alias(new: &str, old: &str) -> Option<String> {
    std::env::var(new).ok().or_else(|| std::env::var(old).ok())
}

pub fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim();
            !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"))
        }
        Err(_) => std::env::var_os(name).is_some(),
    }
}

pub fn env_flag_alias(new: &str, old: &str) -> bool {
    env_flag(new) || env_flag(old)
}

/// Serialize `BUCKYBOI_MODELS` mutations across parallel tests.
#[cfg(test)]
pub(crate) fn with_models_dir<R>(dir: &std::path::Path, f: impl FnOnce() -> R) -> R {
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let prev = std::env::var("BUCKYBOI_MODELS").ok();
    std::env::set_var("BUCKYBOI_MODELS", dir);
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            match &self.0 {
                Some(p) => std::env::set_var("BUCKYBOI_MODELS", p),
                None => std::env::remove_var("BUCKYBOI_MODELS"),
            }
        }
    }
    let _restore = Restore(prev);
    f()
}
