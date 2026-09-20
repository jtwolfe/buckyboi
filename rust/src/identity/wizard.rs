//! First-run / add-person enroll wizard (pure FSM).
//!
//! Sequences name → face → voice → optional hands by driving the same
//! [`EnrollSession`] machines Settings → ID uses. Skip / cancel / complete
//! are unit-tested without a camera or mic.

use crate::identity::enroll::{EnrollKind, EnrollPhase, EnrollSession};
use crate::identity::gate::GateMode;
use crate::identity::hands::GestureClass;

/// Chip when SCRFD (or the detector) sees more than one face.
pub const LARGEST_FACE_CHIP: &str = "USING LARGEST FACE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WizardStep {
    Closed,
    Name,
    Face,
    Voice,
    Hands { class: GestureClass },
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WizardEvent {
    None,
    Opened,
    BeginFace,
    BeginVoice,
    BeginHands(GestureClass),
    SuggestGate(GateMode),
    Cancelled,
    Completed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FirstRunWizard {
    pub step: WizardStep,
    pub person_name: String,
    pub person_id: Option<String>,
    pub face_ok: bool,
    pub voice_ok: bool,
    pub hands_ok: bool,
    pub first_run: bool,
    pub faces_in_frame: u32,
    pub suggested_gate: Option<GateMode>,
}

impl Default for FirstRunWizard {
    fn default() -> Self {
        Self::closed()
    }
}

impl FirstRunWizard {
    pub fn closed() -> Self {
        Self {
            step: WizardStep::Closed,
            person_name: String::new(),
            person_id: None,
            face_ok: false,
            voice_ok: false,
            hands_ok: false,
            first_run: false,
            faces_in_frame: 0,
            suggested_gate: None,
        }
    }

    pub fn is_open(&self) -> bool {
        !matches!(self.step, WizardStep::Closed)
    }

    /// Gate on + empty gallery → day-one wizard (Omarchy / first listen).
    pub fn should_offer(gate: GateMode, people: usize) -> bool {
        gate != GateMode::Off && people == 0
    }

    pub fn open(name: impl Into<String>, first_run: bool) -> Self {
        let mut w = Self::closed();
        w.step = WizardStep::Name;
        w.person_name = {
            let n = name.into();
            if n.trim().is_empty() {
                "P1".into()
            } else {
                n
            }
        };
        w.first_run = first_run;
        w
    }

    /// Rename on the Name step (default P1; editable when a UI can type).
    pub fn set_name(&mut self, name: impl Into<String>) {
        if matches!(self.step, WizardStep::Name) {
            let n = name.into();
            if !n.trim().is_empty() {
                self.person_name = n;
            }
        }
    }

    pub fn note_faces(&mut self, n: u32) {
        self.faces_in_frame = n;
    }

    pub fn using_largest_face(&self) -> bool {
        self.is_open() && self.faces_in_frame > 1
    }

    pub fn face_chip(&self) -> Option<&'static str> {
        if self.using_largest_face() {
            Some(LARGEST_FACE_CHIP)
        } else {
            None
        }
    }

    pub fn can_skip(&self) -> bool {
        matches!(
            self.step,
            WizardStep::Face | WizardStep::Voice | WizardStep::Hands { .. }
        )
    }

    pub fn can_next(&self) -> bool {
        matches!(self.step, WizardStep::Name | WizardStep::Done)
    }

    pub fn can_add_another(&self) -> bool {
        matches!(self.step, WizardStep::Done)
    }

    pub fn title(&self) -> &'static str {
        match self.step {
            WizardStep::Closed => "",
            WizardStep::Name => {
                if self.first_run {
                    "FIRST RUN"
                } else {
                    "ADD PERSON"
                }
            }
            WizardStep::Face => "FACE",
            WizardStep::Voice => "VOICE",
            WizardStep::Hands { .. } => "HANDS",
            WizardStep::Done => "DONE",
        }
    }

    pub fn body(&self, enroll: &EnrollSession) -> String {
        match self.step {
            WizardStep::Closed => String::new(),
            WizardStep::Name => self.person_name.clone(),
            WizardStep::Face | WizardStep::Voice | WizardStep::Hands { .. } => {
                let h = enroll.hint();
                if h.is_empty() {
                    match self.step {
                        WizardStep::Face => "LOOK AT CAM".into(),
                        WizardStep::Voice => "SAY PHRASE".into(),
                        WizardStep::Hands { class } => format!("HOLD {}", class.label()),
                        _ => String::new(),
                    }
                } else {
                    h
                }
            }
            WizardStep::Done => {
                let mut bits = Vec::new();
                if self.face_ok {
                    bits.push("FACE");
                }
                if self.voice_ok {
                    bits.push("VOICE");
                }
                if self.hands_ok {
                    bits.push("HANDS");
                }
                if bits.is_empty() {
                    format!("{}", self.person_name)
                } else {
                    format!("{} {}", self.person_name, bits.join(" "))
                }
            }
        }
    }

    pub fn progress(&self, enroll: &EnrollSession) -> f32 {
        match self.step {
            WizardStep::Done => 1.0,
            WizardStep::Face | WizardStep::Voice | WizardStep::Hands { .. } => enroll.progress(),
            _ => 0.0,
        }
    }

    pub fn suggested_gate_for_enrolled(&self) -> Option<GateMode> {
        match (self.face_ok, self.voice_ok) {
            (true, true) => Some(GateMode::Any),
            (true, false) => Some(GateMode::Face),
            (false, true) => Some(GateMode::Voice),
            (false, false) => None,
        }
    }

    /// Confirm name / finish Done.
    pub fn advance(&mut self) -> WizardEvent {
        match self.step {
            WizardStep::Name => self.enter_face(),
            WizardStep::Done => self.finish(),
            _ => WizardEvent::None,
        }
    }

    /// Skip the current optional modality (voice / remaining hands; face too
    /// so a VM without a camera can still complete).
    pub fn skip(&mut self) -> WizardEvent {
        match self.step {
            WizardStep::Face => self.enter_voice(),
            WizardStep::Voice => self.enter_hands(GestureClass::Fist),
            WizardStep::Hands { .. } => self.enter_done(),
            _ => WizardEvent::None,
        }
    }

    pub fn cancel(&mut self) -> WizardEvent {
        if !self.is_open() {
            return WizardEvent::None;
        }
        *self = Self::closed();
        WizardEvent::Cancelled
    }

    pub fn complete(&mut self) -> WizardEvent {
        if matches!(self.step, WizardStep::Done) {
            self.finish()
        } else {
            WizardEvent::None
        }
    }

    /// After Done, start another person (P2, …).
    pub fn add_another(&mut self, name: impl Into<String>) -> WizardEvent {
        if !matches!(self.step, WizardStep::Done) {
            return WizardEvent::None;
        }
        let first = self.first_run;
        *self = Self::open(name, false);
        self.first_run = first && false;
        WizardEvent::Opened
    }

    /// Settings → ID enroll machines report Finished here.
    pub fn on_enroll_finished(&mut self, kind: EnrollKind) -> WizardEvent {
        match (self.step, kind) {
            (WizardStep::Face, EnrollKind::Face) => {
                self.face_ok = true;
                self.enter_voice()
            }
            (WizardStep::Voice, EnrollKind::Voice) => {
                self.voice_ok = true;
                self.enter_hands(GestureClass::Fist)
            }
            (WizardStep::Hands { class }, EnrollKind::Gesture) => {
                self.hands_ok = true;
                match class.next_train() {
                    Some(next) => self.enter_hands(next),
                    None => self.enter_done(),
                }
            }
            _ => WizardEvent::None,
        }
    }

    fn enter_face(&mut self) -> WizardEvent {
        self.step = WizardStep::Face;
        WizardEvent::BeginFace
    }

    fn enter_voice(&mut self) -> WizardEvent {
        self.step = WizardStep::Voice;
        WizardEvent::BeginVoice
    }

    fn enter_hands(&mut self, class: GestureClass) -> WizardEvent {
        self.step = WizardStep::Hands { class };
        WizardEvent::BeginHands(class)
    }

    fn enter_done(&mut self) -> WizardEvent {
        self.step = WizardStep::Done;
        self.suggested_gate = self.suggested_gate_for_enrolled();
        WizardEvent::None
    }

    fn finish(&mut self) -> WizardEvent {
        let gate = self
            .suggested_gate
            .or_else(|| self.suggested_gate_for_enrolled());
        *self = Self::closed();
        if let Some(g) = gate {
            return WizardEvent::SuggestGate(g);
        }
        WizardEvent::Completed
    }
}

/// True while a first-run / add-person wizard is capturing, so hide-timer
/// and Esc treat it like Settings.
pub fn wizard_blocks_hide(wizard: &FirstRunWizard, enroll: &EnrollSession) -> bool {
    wizard.is_open() || !matches!(enroll.phase, EnrollPhase::Idle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::enroll::EnrollSession;

    #[test]
    fn should_offer_only_when_gate_on_and_empty() {
        assert!(!FirstRunWizard::should_offer(GateMode::Off, 0));
        assert!(FirstRunWizard::should_offer(GateMode::Face, 0));
        assert!(FirstRunWizard::should_offer(GateMode::Any, 0));
        assert!(!FirstRunWizard::should_offer(GateMode::Face, 1));
    }

    #[test]
    fn starts_on_name_default_p1() {
        let w = FirstRunWizard::open("P1", true);
        assert_eq!(w.step, WizardStep::Name);
        assert_eq!(w.person_name, "P1");
        assert!(w.first_run);
        assert!(w.can_next());
        assert!(!w.can_skip());
        assert_eq!(w.title(), "FIRST RUN");
    }

    #[test]
    fn set_name_is_editable_on_name_step() {
        let mut w = FirstRunWizard::open("P1", true);
        w.set_name("Ada");
        assert_eq!(w.person_name, "Ada");
        w.advance();
        w.set_name("Bo");
        assert_eq!(w.person_name, "Ada");
    }

    #[test]
    fn advance_name_begins_face() {
        let mut w = FirstRunWizard::open("P1", true);
        assert_eq!(w.advance(), WizardEvent::BeginFace);
        assert_eq!(w.step, WizardStep::Face);
        assert!(w.can_skip());
        let enroll = EnrollSession::start_face("P1", None);
        assert_eq!(w.body(&enroll), "LOOK AT CAM");
    }

    #[test]
    fn skip_voice_and_hands_reaches_done() {
        let mut w = FirstRunWizard::open("P1", true);
        assert_eq!(w.advance(), WizardEvent::BeginFace);
        assert_eq!(w.skip(), WizardEvent::BeginVoice);
        assert_eq!(w.step, WizardStep::Voice);
        assert_eq!(w.skip(), WizardEvent::BeginHands(GestureClass::Fist));
        assert_eq!(
            w.step,
            WizardStep::Hands {
                class: GestureClass::Fist
            }
        );
        assert_eq!(w.skip(), WizardEvent::None);
        assert_eq!(w.step, WizardStep::Done);
        assert!(w.can_add_another());
        assert!(w.suggested_gate.is_none());
    }

    #[test]
    fn cancel_closes_without_complete() {
        let mut w = FirstRunWizard::open("P1", true);
        w.advance();
        assert_eq!(w.cancel(), WizardEvent::Cancelled);
        assert_eq!(w.step, WizardStep::Closed);
        assert!(!w.is_open());
        assert_eq!(w.cancel(), WizardEvent::None);
    }

    #[test]
    fn face_finish_advances_to_voice() {
        let mut w = FirstRunWizard::open("Ada", true);
        w.advance();
        assert_eq!(
            w.on_enroll_finished(EnrollKind::Face),
            WizardEvent::BeginVoice
        );
        assert!(w.face_ok);
        assert_eq!(w.step, WizardStep::Voice);
    }

    #[test]
    fn complete_suggests_face_gate() {
        let mut w = FirstRunWizard::open("Ada", true);
        w.advance();
        let _ = w.on_enroll_finished(EnrollKind::Face);
        let _ = w.skip(); // skip voice → hands
        let _ = w.skip(); // skip hands → done
        assert_eq!(w.suggested_gate, Some(GateMode::Face));
        assert_eq!(w.complete(), WizardEvent::SuggestGate(GateMode::Face));
        assert_eq!(w.step, WizardStep::Closed);
    }

    #[test]
    fn complete_suggests_any_when_face_and_voice() {
        let mut w = FirstRunWizard::open("Ada", true);
        w.advance();
        let _ = w.on_enroll_finished(EnrollKind::Face);
        let _ = w.on_enroll_finished(EnrollKind::Voice);
        let _ = w.skip(); // skip hands
        assert_eq!(w.suggested_gate, Some(GateMode::Any));
        assert_eq!(w.advance(), WizardEvent::SuggestGate(GateMode::Any));
    }

    #[test]
    fn add_another_restarts_name() {
        let mut w = FirstRunWizard::open("P1", true);
        w.advance();
        let _ = w.on_enroll_finished(EnrollKind::Face);
        let _ = w.skip();
        let _ = w.skip();
        assert_eq!(w.step, WizardStep::Done);
        assert_eq!(w.add_another("P2"), WizardEvent::Opened);
        assert_eq!(w.step, WizardStep::Name);
        assert_eq!(w.person_name, "P2");
        assert!(!w.face_ok);
        assert_eq!(w.title(), "ADD PERSON");
    }

    #[test]
    fn hands_sequence_fist_to_peace() {
        let mut w = FirstRunWizard::open("Ada", false);
        w.advance();
        let _ = w.skip(); // face
        let _ = w.skip(); // voice
        assert_eq!(
            w.step,
            WizardStep::Hands {
                class: GestureClass::Fist
            }
        );
        assert_eq!(
            w.on_enroll_finished(EnrollKind::Gesture),
            WizardEvent::BeginHands(GestureClass::Palm)
        );
        assert_eq!(
            w.on_enroll_finished(EnrollKind::Gesture),
            WizardEvent::BeginHands(GestureClass::ThumbsUp)
        );
        assert_eq!(
            w.on_enroll_finished(EnrollKind::Gesture),
            WizardEvent::BeginHands(GestureClass::Point)
        );
        assert_eq!(
            w.on_enroll_finished(EnrollKind::Gesture),
            WizardEvent::BeginHands(GestureClass::Peace)
        );
        assert_eq!(w.on_enroll_finished(EnrollKind::Gesture), WizardEvent::None);
        assert_eq!(w.step, WizardStep::Done);
        assert!(w.hands_ok);
    }

    #[test]
    fn using_largest_face_when_multiple() {
        let mut w = FirstRunWizard::open("P1", true);
        w.advance();
        w.note_faces(1);
        assert!(!w.using_largest_face());
        w.note_faces(2);
        assert!(w.using_largest_face());
        assert_eq!(w.face_chip(), Some(LARGEST_FACE_CHIP));
    }

    #[test]
    fn skip_on_name_is_noop() {
        let mut w = FirstRunWizard::open("P1", true);
        assert_eq!(w.skip(), WizardEvent::None);
        assert_eq!(w.step, WizardStep::Name);
    }

    #[test]
    fn complete_before_done_is_noop() {
        let mut w = FirstRunWizard::open("P1", true);
        assert_eq!(w.complete(), WizardEvent::None);
        w.advance();
        assert_eq!(w.complete(), WizardEvent::None);
    }

    #[test]
    fn wizard_blocks_hide_while_open() {
        let w = FirstRunWizard::open("P1", true);
        let enroll = EnrollSession::idle();
        assert!(wizard_blocks_hide(&w, &enroll));
        let closed = FirstRunWizard::closed();
        assert!(!wizard_blocks_hide(&closed, &enroll));
    }
}
