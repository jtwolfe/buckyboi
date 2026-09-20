//! Enrollment / calibration state machines (pure, no camera or mic).

use crate::identity::embed::Embedding;
use crate::identity::face::{FaceQuality, FaceReject, FACE_ENROLL_NEED};
use crate::identity::hands::{GestureClass, HandStatus};
use crate::identity::voice::{
    VoiceQuality, VoiceReject, VOICE_ENROLL_NEED, VOICE_ENROLL_REJECT_CAP,
};

/// Continuous palm-miss ticks before gesture enroll fails (~6.4 s at 80 ms).
pub const GESTURE_NO_HAND_CAP: u32 = 80;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnrollKind {
    Face,
    Voice,
    Gesture,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EnrollPhase {
    Idle,
    Prompt {
        kind: EnrollKind,
        person: String,
        gesture: Option<GestureClass>,
    },
    Capturing {
        kind: EnrollKind,
        person: String,
        have: usize,
        need: usize,
        gesture: Option<GestureClass>,
    },
    Done {
        person: String,
        kind: EnrollKind,
        accepted: usize,
    },
    Failed {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnrollEvent {
    None,
    Accepted,
    Rejected,
    Finished,
    Failed,
}

#[derive(Clone, Debug)]
pub struct EnrollSession {
    pub phase: EnrollPhase,
    pub person_name: String,
    pub person_id: Option<String>,
    pub kind: EnrollKind,
    pub gesture: Option<GestureClass>,
    pub accepted: Vec<Embedding>,
    pub gesture_samples: Vec<Vec<f32>>,
    pub rejects: u32,
    pub need: usize,
    pub last_reject: Option<String>,
    /// Wall clock origin for voice timeout. Stamped once by `begin_capture_at`.
    pub started_ms: u64,
}

impl EnrollSession {
    pub fn idle() -> Self {
        Self {
            phase: EnrollPhase::Idle,
            person_name: String::new(),
            person_id: None,
            kind: EnrollKind::Face,
            gesture: None,
            accepted: Vec::new(),
            gesture_samples: Vec::new(),
            rejects: 0,
            need: FACE_ENROLL_NEED,
            last_reject: None,
            started_ms: 0,
        }
    }

    pub fn start_face(name: impl Into<String>, person_id: Option<String>) -> Self {
        let person_name = name.into();
        Self {
            phase: EnrollPhase::Prompt {
                kind: EnrollKind::Face,
                person: person_name.clone(),
                gesture: None,
            },
            person_name,
            person_id,
            kind: EnrollKind::Face,
            gesture: None,
            accepted: Vec::new(),
            gesture_samples: Vec::new(),
            rejects: 0,
            need: FACE_ENROLL_NEED,
            last_reject: None,
            started_ms: 0,
        }
    }

    pub fn start_voice(name: impl Into<String>, person_id: Option<String>) -> Self {
        let person_name = name.into();
        Self {
            phase: EnrollPhase::Prompt {
                kind: EnrollKind::Voice,
                person: person_name.clone(),
                gesture: None,
            },
            person_name,
            person_id,
            kind: EnrollKind::Voice,
            gesture: None,
            accepted: Vec::new(),
            gesture_samples: Vec::new(),
            rejects: 0,
            need: VOICE_ENROLL_NEED,
            last_reject: None,
            started_ms: 0,
        }
    }

    pub fn start_gesture(
        name: impl Into<String>,
        person_id: Option<String>,
        class: GestureClass,
        need: usize,
    ) -> Self {
        let person_name = name.into();
        Self {
            phase: EnrollPhase::Prompt {
                kind: EnrollKind::Gesture,
                person: person_name.clone(),
                gesture: Some(class),
            },
            person_name,
            person_id,
            kind: EnrollKind::Gesture,
            gesture: Some(class),
            accepted: Vec::new(),
            gesture_samples: Vec::new(),
            rejects: 0,
            need: need.max(3),
            last_reject: None,
            started_ms: 0,
        }
    }

    /// Phase → Capturing only. Does not move `started_ms` (timeout origin).
    pub fn begin_capture(&mut self) {
        if matches!(
            self.phase,
            EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. }
        ) {
            return;
        }
        self.phase = EnrollPhase::Capturing {
            kind: self.kind,
            person: self.person_name.clone(),
            have: self.accepted.len().max(self.gesture_samples.len()),
            need: self.need,
            gesture: self.gesture,
        };
    }

    /// Call once from StartEnroll / wizard Begin. Idempotent: stamps only if 0.
    pub fn begin_capture_at(&mut self, now_ms: u64) {
        self.begin_capture();
        if self.started_ms == 0 {
            self.started_ms = now_ms;
        }
    }

    pub fn timed_out(&self, now: u64, limit: u64) -> bool {
        self.started_ms > 0 && now.saturating_sub(self.started_ms) >= limit
    }

    pub fn cancel(&mut self) {
        *self = Self::idle();
    }

    /// Immediate `Failed`. Missing palm ONNX (`NO MODEL`) uses this so the
    /// wizard can skip instead of waiting on the reject cap.
    pub fn fail_now(&mut self, reason: impl Into<String>) -> EnrollEvent {
        if matches!(
            self.phase,
            EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. }
        ) {
            return EnrollEvent::None;
        }
        let reason = reason.into();
        if !matches!(&self.phase, EnrollPhase::Failed { reason: r } if r == &reason) {
            eprintln!("buckyboi: enroll failed ({reason})");
        }
        self.last_reject = Some(reason.clone());
        self.phase = EnrollPhase::Failed { reason };
        EnrollEvent::Failed
    }

    /// HUD reject (`NO HAND`). Caps at [`GESTURE_NO_HAND_CAP`] ticks; `Ok`
    /// must zero `rejects` so a flicker does not burn fist enroll.
    pub fn note_reject(&mut self, reason: &str) -> EnrollEvent {
        if matches!(
            self.phase,
            EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. }
        ) {
            return EnrollEvent::None;
        }
        self.begin_capture();
        self.rejects = self.rejects.saturating_add(1);
        self.last_reject = Some(reason.to_string());
        if self.rejects >= GESTURE_NO_HAND_CAP {
            self.phase = EnrollPhase::Failed {
                reason: reason.to_string(),
            };
            return EnrollEvent::Failed;
        }
        EnrollEvent::Rejected
    }

    /// Present-thread consume of a `HandSnap` during gesture enroll.
    pub fn apply_hand_status(&mut self, status: &HandStatus) -> EnrollEvent {
        if !matches!(self.kind, EnrollKind::Gesture) {
            return EnrollEvent::None;
        }
        match status {
            HandStatus::NoModel => self.fail_now("NO MODEL"),
            HandStatus::NoHand => self.note_reject("NO HAND"),
            HandStatus::Ok(h) => self.push_gesture(&h.normalized()),
        }
    }

    pub fn push_face(&mut self, quality: FaceQuality, emb: Option<Embedding>) -> EnrollEvent {
        if !matches!(self.kind, EnrollKind::Face) {
            return EnrollEvent::None;
        }
        self.begin_capture();
        // Missing ArcFace is `NoEmbed` (ok or not). Rec-skip snaps are
        // ok + reject=None + no embedding — ignore, do not fail_now.
        if quality.reject == FaceReject::NoEmbed {
            return self.fail_now("NO MODEL");
        }
        if !quality.ok {
            self.note_reject(quality.reject.hint());
            if self.rejects > 40 {
                self.fail_now(self.last_reject.clone().unwrap_or_else(|| "NO FACE".into()));
                return EnrollEvent::Failed;
            }
            return EnrollEvent::Rejected;
        }
        let Some(emb) = emb else {
            return EnrollEvent::None;
        };
        self.last_reject = None;
        self.accepted.push(emb);
        self.sync_capturing();
        if self.accepted.len() >= self.need {
            self.phase = EnrollPhase::Done {
                person: self.person_name.clone(),
                kind: EnrollKind::Face,
                accepted: self.accepted.len(),
            };
            return EnrollEvent::Finished;
        }
        EnrollEvent::Accepted
    }

    pub fn push_voice(&mut self, quality: VoiceQuality, emb: Option<Embedding>) -> EnrollEvent {
        if !matches!(self.kind, EnrollKind::Voice) {
            return EnrollEvent::None;
        }
        self.begin_capture();
        if quality.reject == VoiceReject::NoMic {
            self.fail_now(quality.reject.hint());
            return EnrollEvent::Failed;
        }
        if !quality.ok {
            self.note_reject(quality.reject.hint());
            if self.rejects >= VOICE_ENROLL_REJECT_CAP {
                self.fail_now(
                    self.last_reject
                        .clone()
                        .unwrap_or_else(|| "TOO QUIET".into()),
                );
                return EnrollEvent::Failed;
            }
            return EnrollEvent::Rejected;
        }
        let Some(emb) = emb else {
            self.last_reject = Some("NO MODEL".into());
            return EnrollEvent::Rejected;
        };
        self.last_reject = None;
        self.accepted.push(emb);
        self.sync_capturing();
        if self.accepted.len() >= self.need {
            self.phase = EnrollPhase::Done {
                person: self.person_name.clone(),
                kind: EnrollKind::Voice,
                accepted: self.accepted.len(),
            };
            return EnrollEvent::Finished;
        }
        EnrollEvent::Accepted
    }

    pub fn push_gesture(&mut self, landmarks: &[f32]) -> EnrollEvent {
        if !matches!(self.kind, EnrollKind::Gesture) {
            return EnrollEvent::None;
        }
        if landmarks.len() < 21 {
            self.rejects = self.rejects.saturating_add(1);
            return EnrollEvent::Rejected;
        }
        self.begin_capture();
        // Flicker must not burn the NO HAND cap; a late fist still enrolls.
        self.rejects = 0;
        self.last_reject = None;
        self.gesture_samples.push(landmarks.to_vec());
        self.sync_capturing();
        if self.gesture_samples.len() >= self.need {
            self.phase = EnrollPhase::Done {
                person: self.person_name.clone(),
                kind: EnrollKind::Gesture,
                accepted: self.gesture_samples.len(),
            };
            return EnrollEvent::Finished;
        }
        EnrollEvent::Accepted
    }

    fn sync_capturing(&mut self) {
        let have = self.accepted.len().max(self.gesture_samples.len());
        self.phase = EnrollPhase::Capturing {
            kind: self.kind,
            person: self.person_name.clone(),
            have,
            need: self.need,
            gesture: self.gesture,
        };
    }

    pub fn progress(&self) -> f32 {
        match &self.phase {
            EnrollPhase::Capturing { have, need, .. } => {
                (*have as f32 / (*need).max(1) as f32).clamp(0.0, 1.0)
            }
            EnrollPhase::Done { .. } => 1.0,
            _ => 0.0,
        }
    }

    pub fn hint(&self) -> String {
        match &self.phase {
            EnrollPhase::Idle => String::new(),
            EnrollPhase::Prompt { kind, gesture, .. } => match kind {
                EnrollKind::Face => "LOOK AT CAM".into(),
                EnrollKind::Voice => "SAY PHRASE".into(),
                EnrollKind::Gesture => {
                    format!("HOLD {}", gesture.unwrap_or(GestureClass::Palm).label())
                }
            },
            EnrollPhase::Capturing {
                kind,
                have,
                need,
                gesture,
                ..
            } => {
                if let Some(r) = &self.last_reject {
                    if !r.is_empty() {
                        return r.clone();
                    }
                }
                match kind {
                    EnrollKind::Face => format!("FACE {have}/{need}"),
                    EnrollKind::Voice => format!("VOICE {have}/{need}"),
                    EnrollKind::Gesture => format!(
                        "{} {have}/{need}",
                        gesture.unwrap_or(GestureClass::Palm).label()
                    ),
                }
            }
            EnrollPhase::Done { kind, .. } => match kind {
                EnrollKind::Face => "FACE SAVED".into(),
                EnrollKind::Voice => "VOICE SAVED".into(),
                EnrollKind::Gesture => "GESTURE SAVED".into(),
            },
            EnrollPhase::Failed { reason } => reason.to_ascii_uppercase(),
        }
    }
}

/// Wizard skip-on-fail. Voice device/timeout/quiet cap and face NO MODEL only.
/// Face `NO FACE` / lighting stays on the 40-reject cap so the user can fix it.
pub fn wizard_auto_skip(kind: EnrollKind, reason: &str) -> bool {
    let r = reason.trim().to_ascii_uppercase();
    match kind {
        EnrollKind::Voice => {
            matches!(r.as_str(), "NO MIC" | "TIMEOUT" | "TOO QUIET" | "TOO SHORT")
        }
        EnrollKind::Face => r == "NO MODEL",
        EnrollKind::Gesture => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::embed::Embedding;
    use crate::identity::face::FaceQuality;
    use crate::identity::voice::{VoiceQuality, VOICE_ENROLL_TIMEOUT_MS};

    fn ok_face() -> FaceQuality {
        FaceQuality {
            area_frac: 0.12,
            brightness: 120.0,
            sharpness: 80.0,
            faces: 1,
            ok: true,
            reject: crate::identity::face::FaceReject::None,
        }
    }

    #[test]
    fn face_enroll_needs_quality_and_n_frames() {
        let mut s = EnrollSession::start_face("Ada", None);
        assert!(matches!(s.phase, EnrollPhase::Prompt { .. }));
        let bad = FaceQuality {
            ok: false,
            ..ok_face()
        };
        assert_eq!(
            s.push_face(bad, Some(Embedding::new("arcface", vec![1.0, 0.0]))),
            EnrollEvent::Rejected
        );
        for i in 0..FACE_ENROLL_NEED {
            let ev = s.push_face(
                ok_face(),
                Some(Embedding::new("arcface", vec![1.0, i as f32 * 0.01])),
            );
            if i + 1 < FACE_ENROLL_NEED {
                assert_eq!(ev, EnrollEvent::Accepted);
            } else {
                assert_eq!(ev, EnrollEvent::Finished);
            }
        }
        assert!(
            matches!(s.phase, EnrollPhase::Done { accepted, .. } if accepted == FACE_ENROLL_NEED)
        );
    }

    #[test]
    fn voice_enroll_rejects_quiet() {
        let mut s = EnrollSession::start_voice("Ada", None);
        let quiet = VoiceQuality {
            duration_ms: 400,
            energy: 0.001,
            ok: false,
            reject: crate::identity::voice::VoiceReject::TooShort,
        };
        assert_eq!(
            s.push_voice(quiet, Some(Embedding::new("logmel", vec![1.0; 8]))),
            EnrollEvent::Rejected
        );
        assert_eq!(s.hint(), "TOO SHORT");
    }

    #[test]
    fn gesture_enroll_finishes() {
        let mut s = EnrollSession::start_gesture("Ada", None, GestureClass::Fist, 3);
        let lm = vec![0.0; 63];
        assert_eq!(s.push_gesture(&lm), EnrollEvent::Accepted);
        assert_eq!(s.push_gesture(&lm), EnrollEvent::Accepted);
        assert_eq!(s.push_gesture(&lm), EnrollEvent::Finished);
    }

    #[test]
    fn cancel_returns_idle() {
        let mut s = EnrollSession::start_face("Ada", None);
        s.cancel();
        assert_eq!(s.phase, EnrollPhase::Idle);
    }

    #[test]
    fn face_reject_shows_reason() {
        let mut s = EnrollSession::start_face("Ada", None);
        let dark = FaceQuality {
            ok: false,
            reject: crate::identity::face::FaceReject::TooDark,
            ..ok_face()
        };
        assert_eq!(s.push_face(dark, None), EnrollEvent::Rejected);
        assert_eq!(s.hint(), "TOO DARK");
    }

    fn quiet_q() -> VoiceQuality {
        VoiceQuality {
            duration_ms: 0,
            energy: 0.0,
            ok: false,
            reject: crate::identity::voice::VoiceReject::TooShort,
        }
    }

    #[test]
    fn no_mic_fails_immediately() {
        let mut s = EnrollSession::start_voice("Ada", None);
        let q = VoiceQuality {
            duration_ms: 0,
            energy: 0.0,
            ok: false,
            reject: crate::identity::voice::VoiceReject::NoMic,
        };
        assert_eq!(s.push_voice(q, None), EnrollEvent::Failed);
        assert!(matches!(s.phase, EnrollPhase::Failed { ref reason } if reason == "NO MIC"));
        assert_eq!(s.hint(), "NO MIC");
        s.fail_now("NO MIC");
        assert!(!wizard_auto_skip(EnrollKind::Face, "NO FACE"));
        assert!(wizard_auto_skip(EnrollKind::Voice, "NO MIC"));
    }

    #[test]
    fn push_voice_rejects_do_not_extend_deadline() {
        let mut s = EnrollSession::start_voice("Ada", None);
        s.begin_capture_at(1_000);
        assert_eq!(s.push_voice(quiet_q(), None), EnrollEvent::Rejected);
        s.begin_capture_at(3_000);
        assert_eq!(s.started_ms, 1_000);
        assert_eq!(s.push_voice(quiet_q(), None), EnrollEvent::Rejected);
        assert_eq!(s.started_ms, 1_000);
        assert!(!s.timed_out(1_000 + VOICE_ENROLL_TIMEOUT_MS - 1, VOICE_ENROLL_TIMEOUT_MS));
        assert!(s.timed_out(1_000 + VOICE_ENROLL_TIMEOUT_MS, VOICE_ENROLL_TIMEOUT_MS));
    }

    #[test]
    fn empty_voice_ticks_increment_rejects() {
        let mut s = EnrollSession::start_voice("Ada", None);
        s.begin_capture_at(0);
        let empty = VoiceQuality::assess(&[], crate::identity::voice::VOICE_SAMPLE_RATE);
        assert!(!empty.ok);
        for i in 0..VOICE_ENROLL_REJECT_CAP - 1 {
            assert_eq!(s.push_voice(empty, None), EnrollEvent::Rejected);
            assert_eq!(s.rejects, i + 1);
        }
        assert_eq!(s.push_voice(empty, None), EnrollEvent::Failed);
        assert!(matches!(s.phase, EnrollPhase::Failed { .. }));
        assert!(wizard_auto_skip(EnrollKind::Voice, &s.hint()));
    }

    #[test]
    fn no_embed_fails_immediately_no_face_does_not_auto_skip() {
        let mut s = EnrollSession::start_face("Ada", None);
        let no_model = FaceQuality {
            ok: false,
            reject: crate::identity::face::FaceReject::NoEmbed,
            ..ok_face()
        };
        assert_eq!(s.push_face(no_model, None), EnrollEvent::Failed);
        assert!(wizard_auto_skip(EnrollKind::Face, "NO MODEL"));
        assert!(!wizard_auto_skip(EnrollKind::Face, "NO FACE"));
        assert!(!wizard_auto_skip(EnrollKind::Face, "TOO DARK"));
        let mut face = EnrollSession::start_face("Ada", None);
        let none = FaceQuality {
            ok: false,
            reject: crate::identity::face::FaceReject::NoFace,
            ..ok_face()
        };
        assert_eq!(face.push_face(none, None), EnrollEvent::Rejected);
        assert!(!matches!(face.phase, EnrollPhase::Failed { .. }));
    }

    #[test]
    fn begin_capture_does_not_move_started_ms() {
        let mut s = EnrollSession::start_voice("Ada", None);
        s.begin_capture_at(50);
        s.begin_capture();
        assert_eq!(s.started_ms, 50);
    }

    #[test]
    fn face_no_embed_fails_now_rec_skip_ignored() {
        let mut s = EnrollSession::start_face("Ada", None);
        let no_model = FaceQuality {
            ok: false,
            reject: FaceReject::NoEmbed,
            ..ok_face()
        };
        assert_eq!(s.push_face(no_model, None), EnrollEvent::Failed);
        assert!(matches!(
            s.phase,
            EnrollPhase::Failed { ref reason } if reason == "NO MODEL"
        ));
        assert_eq!(s.hint(), "NO MODEL");

        let mut skip = EnrollSession::start_face("Ada", None);
        let rec_skip = FaceQuality {
            ok: true,
            reject: FaceReject::None,
            ..ok_face()
        };
        assert_eq!(skip.push_face(rec_skip, None), EnrollEvent::None);
        assert!(!matches!(skip.phase, EnrollPhase::Failed { .. }));
        assert_eq!(skip.rejects, 0);
        assert!(skip.accepted.is_empty());

        let mut t = EnrollSession::start_face("Ada", None);
        let no_face = FaceQuality {
            ok: false,
            reject: FaceReject::NoFace,
            ..ok_face()
        };
        assert_eq!(t.push_face(no_face, None), EnrollEvent::Rejected);
        assert!(!matches!(t.phase, EnrollPhase::Failed { .. }));
        assert_eq!(t.hint(), "NO FACE");
        assert_eq!(t.rejects, 1);
        let dark = FaceQuality {
            ok: false,
            reject: FaceReject::TooDark,
            ..ok_face()
        };
        assert_eq!(t.push_face(dark, None), EnrollEvent::Rejected);
        assert_eq!(t.rejects, 2);
        assert!(!matches!(t.phase, EnrollPhase::Failed { .. }));
    }

    #[test]
    fn gesture_fail_now_no_model() {
        let mut s = EnrollSession::start_gesture("Ada", None, GestureClass::Fist, 6);
        assert_eq!(s.fail_now("NO MODEL"), EnrollEvent::Failed);
        assert!(matches!(
            s.phase,
            EnrollPhase::Failed { ref reason } if reason == "NO MODEL"
        ));
        assert_eq!(s.hint(), "NO MODEL");
        assert!(s.hint().len() <= 16);
        assert_eq!(s.fail_now("NO MODEL"), EnrollEvent::None);
    }

    #[test]
    fn fail_now_idle_is_noop() {
        let mut s = EnrollSession::idle();
        assert_eq!(s.fail_now("NO MODEL"), EnrollEvent::None);
        assert_eq!(s.phase, EnrollPhase::Idle);
    }

    #[test]
    fn gesture_note_reject_sets_hint_and_caps() {
        let mut s = EnrollSession::start_gesture("Ada", None, GestureClass::Fist, 6);
        assert_eq!(s.note_reject("NO HAND"), EnrollEvent::Rejected);
        assert_eq!(s.last_reject.as_deref(), Some("NO HAND"));
        assert_eq!(s.hint(), "NO HAND");
        assert!(s.hint().len() <= 16);
        for i in 2..GESTURE_NO_HAND_CAP {
            assert_eq!(s.note_reject("NO HAND"), EnrollEvent::Rejected, "tick {i}");
        }
        assert_eq!(s.rejects, GESTURE_NO_HAND_CAP - 1);
        assert_eq!(s.note_reject("NO HAND"), EnrollEvent::Failed);
        assert!(matches!(
            s.phase,
            EnrollPhase::Failed { ref reason } if reason == "NO HAND"
        ));
        assert_eq!(s.hint(), "NO HAND");
    }

    #[test]
    fn gesture_note_reject_resets_on_ok() {
        let mut s = EnrollSession::start_gesture("Ada", None, GestureClass::Fist, 6);
        for _ in 0..10 {
            assert_eq!(s.note_reject("NO HAND"), EnrollEvent::Rejected);
        }
        assert_eq!(s.rejects, 10);
        let lm = vec![0.0; 42];
        assert_eq!(s.push_gesture(&lm), EnrollEvent::Accepted);
        assert_eq!(s.rejects, 0);
        assert!(s.last_reject.is_none());
        assert_eq!(s.hint(), "FIST 1/6");
        assert!(s.hint().len() <= 16);
        for _ in 0..(GESTURE_NO_HAND_CAP - 1) {
            assert_eq!(s.note_reject("NO HAND"), EnrollEvent::Rejected);
        }
        assert_eq!(s.note_reject("NO HAND"), EnrollEvent::Failed);
    }

    #[test]
    fn apply_hand_status_no_model_no_hand_ok() {
        let mut s = EnrollSession::start_gesture("Ada", None, GestureClass::Fist, 6);
        assert_eq!(
            s.apply_hand_status(&HandStatus::NoHand),
            EnrollEvent::Rejected
        );
        assert_eq!(s.hint(), "NO HAND");
        let hand = crate::identity::hands::synthetic(GestureClass::Fist);
        assert_eq!(
            s.apply_hand_status(&HandStatus::Ok(hand)),
            EnrollEvent::Accepted
        );
        assert_eq!(s.rejects, 0);
        assert_eq!(s.hint(), "FIST 1/6");
        let mut t = EnrollSession::start_gesture("Ada", None, GestureClass::Fist, 6);
        assert_eq!(
            t.apply_hand_status(&HandStatus::NoModel),
            EnrollEvent::Failed
        );
        assert_eq!(t.hint(), "NO MODEL");
    }
}
