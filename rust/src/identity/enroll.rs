//! Enrollment / calibration state machines (pure, no camera or mic).

use crate::identity::embed::Embedding;
use crate::identity::face::{FaceQuality, FACE_ENROLL_NEED};
use crate::identity::hands::GestureClass;
use crate::identity::voice::{VoiceQuality, VOICE_ENROLL_NEED};

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
        }
    }

    pub fn begin_capture(&mut self) {
        if matches!(self.phase, EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. }) {
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

    pub fn cancel(&mut self) {
        *self = Self::idle();
    }

    pub fn push_face(&mut self, quality: FaceQuality, emb: Option<Embedding>) -> EnrollEvent {
        if !matches!(self.kind, EnrollKind::Face) {
            return EnrollEvent::None;
        }
        self.begin_capture();
        if !quality.ok {
            self.rejects = self.rejects.saturating_add(1);
            self.last_reject = Some(quality.reject.hint().into());
            if self.rejects > 40 {
                self.phase = EnrollPhase::Failed {
                    reason: self
                        .last_reject
                        .clone()
                        .unwrap_or_else(|| "NO FACE".into()),
                };
                return EnrollEvent::Failed;
            }
            return EnrollEvent::Rejected;
        }
        let Some(emb) = emb else {
            self.rejects = self.rejects.saturating_add(1);
            self.last_reject = Some(crate::identity::face::FaceReject::NoEmbed.hint().into());
            return EnrollEvent::Rejected;
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
        if !quality.ok {
            self.rejects = self.rejects.saturating_add(1);
            self.last_reject = Some(quality.reject.hint().into());
            if self.rejects > 20 {
                self.phase = EnrollPhase::Failed {
                    reason: self
                        .last_reject
                        .clone()
                        .unwrap_or_else(|| "TOO SHORT".into()),
                };
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
                EnrollKind::Gesture => format!("HOLD {}", gesture.unwrap_or(GestureClass::Palm).label()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::embed::Embedding;
    use crate::identity::face::FaceQuality;
    use crate::identity::voice::VoiceQuality;

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
        assert_eq!(s.push_face(bad, Some(Embedding::new("arcface", vec![1.0, 0.0]))), EnrollEvent::Rejected);
        for i in 0..FACE_ENROLL_NEED {
            let ev = s.push_face(ok_face(), Some(Embedding::new("arcface", vec![1.0, i as f32 * 0.01])));
            if i + 1 < FACE_ENROLL_NEED {
                assert_eq!(ev, EnrollEvent::Accepted);
            } else {
                assert_eq!(ev, EnrollEvent::Finished);
            }
        }
        assert!(matches!(s.phase, EnrollPhase::Done { accepted, .. } if accepted == FACE_ENROLL_NEED));
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
        assert_eq!(s.push_voice(quiet, Some(Embedding::new("logmel", vec![1.0; 8]))), EnrollEvent::Rejected);
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
}
