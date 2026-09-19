//! Who may drive listen / sensitive actions.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GateMode {
    /// Anyone can listen (default so a fresh install still works).
    #[default]
    Off,
    Face,
    Voice,
    /// Face or voice (whichever matches first).
    Any,
    /// Face and voice must agree on the same person.
    All,
}

impl GateMode {
    pub fn as_str(self) -> &'static str {
        match self {
            GateMode::Off => "off",
            GateMode::Face => "face",
            GateMode::Voice => "voice",
            GateMode::Any => "any",
            GateMode::All => "all",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "face" => GateMode::Face,
            "voice" => GateMode::Voice,
            "any" => GateMode::Any,
            "all" | "both" => GateMode::All,
            _ => GateMode::Off,
        }
    }

    pub fn next(self) -> Self {
        match self {
            GateMode::Off => GateMode::Face,
            GateMode::Face => GateMode::Voice,
            GateMode::Voice => GateMode::Any,
            GateMode::Any => GateMode::All,
            GateMode::All => GateMode::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            GateMode::Off => "OFF",
            GateMode::Face => "FACE",
            GateMode::Voice => "VOICE",
            GateMode::Any => "ANY",
            GateMode::All => "ALL",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthVia {
    None,
    Face,
    Voice,
    Both,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct AuthState {
    pub person_id: Option<String>,
    pub name: Option<String>,
    pub via: AuthVia,
    pub until_ms: u64,
    pub score: f32,
}

impl Default for AuthVia {
    fn default() -> Self {
        AuthVia::None
    }
}

impl AuthState {
    pub fn unknown() -> Self {
        Self {
            person_id: None,
            name: None,
            via: AuthVia::None,
            until_ms: 0,
            score: 0.0,
        }
    }

    pub fn matched(id: impl Into<String>, name: impl Into<String>, via: AuthVia, until_ms: u64, score: f32) -> Self {
        Self {
            person_id: Some(id.into()),
            name: Some(name.into()),
            via,
            until_ms,
            score,
        }
    }

    pub fn is_matched(&self, now_ms: u64) -> bool {
        self.person_id.is_some() && now_ms < self.until_ms
    }

    pub fn display_name(&self, now_ms: u64) -> &'static str {
        // Used when we only need a short HUD token; callers prefer `name`.
        let _ = now_ms;
        "AUTH"
    }

    pub fn label(&self, now_ms: u64) -> String {
        if self.is_matched(now_ms) {
            self.name.clone().unwrap_or_else(|| "?".into())
        } else {
            "UNKNOWN".into()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AuthSession {
    pub state: AuthState,
    pub face_id: Option<String>,
    pub face_name: Option<String>,
    pub face_until: u64,
    pub voice_id: Option<String>,
    pub voice_name: Option<String>,
    pub voice_until: u64,
    pub hold_ms: u64,
}

impl AuthSession {
    pub fn new(hold_ms: u64) -> Self {
        Self {
            state: AuthState::unknown(),
            hold_ms: hold_ms.max(500),
            ..Self::default()
        }
    }

    pub fn note_face(&mut self, id: &str, name: &str, now_ms: u64, score: f32) {
        self.face_id = Some(id.to_string());
        self.face_name = Some(name.to_string());
        self.face_until = now_ms + self.hold_ms;
        let _ = score;
        self.recompute(now_ms);
        if let Some(s) = &mut self.state.person_id {
            if s == id {
                self.state.score = self.state.score.max(score);
            }
        }
    }

    pub fn note_voice(&mut self, id: &str, name: &str, now_ms: u64, score: f32) {
        self.voice_id = Some(id.to_string());
        self.voice_name = Some(name.to_string());
        self.voice_until = now_ms + self.hold_ms;
        let _ = score;
        self.recompute(now_ms);
        if let Some(s) = &mut self.state.person_id {
            if s == id {
                self.state.score = self.state.score.max(score);
            }
        }
    }

    pub fn expire(&mut self, now_ms: u64) {
        if self.face_id.is_some() && now_ms >= self.face_until {
            self.face_id = None;
            self.face_name = None;
        }
        if self.voice_id.is_some() && now_ms >= self.voice_until {
            self.voice_id = None;
            self.voice_name = None;
        }
        self.recompute(now_ms);
    }

    fn recompute(&mut self, now_ms: u64) {
        let face_live = self.face_id.as_ref().filter(|_| now_ms < self.face_until);
        let voice_live = self.voice_id.as_ref().filter(|_| now_ms < self.voice_until);
        self.state = match (face_live, voice_live) {
            (Some(f), Some(v)) if f == v => AuthState::matched(
                f.clone(),
                self.face_name.clone().unwrap_or_default(),
                AuthVia::Both,
                self.face_until.max(self.voice_until),
                self.state.score,
            ),
            (Some(f), Some(_)) => AuthState::matched(
                f.clone(),
                self.face_name.clone().unwrap_or_default(),
                AuthVia::Face,
                self.face_until,
                self.state.score,
            ),
            (Some(f), None) => AuthState::matched(
                f.clone(),
                self.face_name.clone().unwrap_or_default(),
                AuthVia::Face,
                self.face_until,
                self.state.score,
            ),
            (None, Some(v)) => AuthState::matched(
                v.clone(),
                self.voice_name.clone().unwrap_or_default(),
                AuthVia::Voice,
                self.voice_until,
                self.state.score,
            ),
            _ => AuthState::unknown(),
        };
    }

    /// Fail closed: unknown never drives listen when a gate is on.
    pub fn allows_listen(&self, mode: GateMode, now_ms: u64) -> bool {
        match mode {
            GateMode::Off => true,
            GateMode::Face => {
                self.face_id.is_some() && now_ms < self.face_until
            }
            GateMode::Voice => {
                self.voice_id.is_some() && now_ms < self.voice_until
            }
            GateMode::Any => {
                (self.face_id.is_some() && now_ms < self.face_until)
                    || (self.voice_id.is_some() && now_ms < self.voice_until)
            }
            GateMode::All => {
                matches!(
                    (
                        self.face_id.as_ref().filter(|_| now_ms < self.face_until),
                        self.voice_id.as_ref().filter(|_| now_ms < self.voice_until),
                    ),
                    (Some(f), Some(v)) if f == v
                )
            }
        }
    }

    pub fn allows_gesture(&self, mode: GateMode, need_face: bool, now_ms: u64) -> bool {
        if need_face {
            return self.face_id.is_some() && now_ms < self.face_until;
        }
        self.allows_listen(mode, now_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_always_allows() {
        let s = AuthSession::new(8_000);
        assert!(s.allows_listen(GateMode::Off, 0));
        assert!(!s.allows_listen(GateMode::Face, 0));
    }

    #[test]
    fn face_gate_needs_live_face() {
        let mut s = AuthSession::new(1_000);
        s.note_face("p1", "Ada", 100, 0.8);
        assert!(s.allows_listen(GateMode::Face, 200));
        assert!(!s.allows_listen(GateMode::Voice, 200));
        s.expire(2_000);
        assert!(!s.allows_listen(GateMode::Face, 2_000));
        assert_eq!(s.state.label(2_000), "UNKNOWN");
    }

    #[test]
    fn all_requires_same_person() {
        let mut s = AuthSession::new(5_000);
        s.note_face("p1", "Ada", 0, 0.9);
        s.note_voice("p2", "Bo", 0, 0.9);
        assert!(!s.allows_listen(GateMode::All, 10));
        s.note_voice("p1", "Ada", 20, 0.7);
        assert!(s.allows_listen(GateMode::All, 30));
        assert_eq!(s.state.via, AuthVia::Both);
    }

    #[test]
    fn any_accepts_voice_alone() {
        let mut s = AuthSession::new(5_000);
        s.note_voice("p1", "Ada", 0, 0.8);
        assert!(s.allows_listen(GateMode::Any, 10));
        assert!(!s.allows_listen(GateMode::Face, 10));
    }

    #[test]
    fn gesture_can_require_prior_face() {
        let mut s = AuthSession::new(5_000);
        s.note_voice("p1", "Ada", 0, 0.9);
        assert!(!s.allows_gesture(GateMode::Any, true, 10));
        s.note_face("p1", "Ada", 10, 0.9);
        assert!(s.allows_gesture(GateMode::Any, true, 20));
    }

    #[test]
    fn gate_cycle() {
        assert_eq!(GateMode::Off.next(), GateMode::Face);
        assert_eq!(GateMode::parse("BOTH"), GateMode::All);
        assert_eq!(GateMode::parse("nope"), GateMode::Off);
    }
}
