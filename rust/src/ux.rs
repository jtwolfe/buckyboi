//! Overlay UX: VisibleIdle / Listening / Hidden, plus tap-vs-drag.
//! See UX.md. No speech / ASR / LLM — this is a visual prototype only.

pub const TAP_MAX_MOVE: f32 = 6.0;
pub const TAP_MAX_MS: u32 = 250;
pub const LISTEN_MIN_MS: u32 = 5_000;
pub const LISTEN_MAX_MS: u32 = 15_000;

pub const UX_VISIBLE: u8 = 0;
pub const UX_LISTENING: u8 = 1;
pub const UX_HIDDEN: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    VisibleIdle,
    Listening,
    Hidden,
}

impl Phase {
    pub fn as_u8(self) -> u8 {
        match self {
            Phase::VisibleIdle => UX_VISIBLE,
            Phase::Listening => UX_LISTENING,
            Phase::Hidden => UX_HIDDEN,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Press {
    pub x: f32,
    pub y: f32,
    pub t_ms: u64,
}

/// Movement < 6px and duration < 250ms counts as a click, not a drag.
pub fn is_tap(dx: f32, dy: f32, dt_ms: u64) -> bool {
    dx.hypot(dy) < TAP_MAX_MOVE && dt_ms < TAP_MAX_MS as u64
}

/// Deterministic 5–15s listen window from a seed (clock or env override).
pub fn listen_duration_ms(seed: u64) -> u32 {
    let span = (LISTEN_MAX_MS - LISTEN_MIN_MS) as u64;
    LISTEN_MIN_MS + (seed % (span + 1)) as u32
}

#[derive(Clone, Debug)]
pub struct BuddyUx {
    pub phase: Phase,
    pub listen_until_ms: u64,
    pub press: Option<Press>,
    pub dragging: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UxEvent {
    None,
    StartListen,
    Hide,
    Wake,
    BeginDrag,
    EndDrag,
}

impl BuddyUx {
    pub fn new() -> Self {
        Self {
            phase: Phase::VisibleIdle,
            listen_until_ms: 0,
            press: None,
            dragging: false,
        }
    }

    pub fn button_down(&mut self, x: f32, y: f32, t_ms: u64, over_hit: bool) -> UxEvent {
        if self.phase != Phase::VisibleIdle || !over_hit {
            return UxEvent::None;
        }
        self.press = Some(Press { x, y, t_ms });
        self.dragging = false;
        UxEvent::None
    }

    pub fn pointer_move(&mut self, x: f32, y: f32, _t_ms: u64) -> UxEvent {
        if self.phase != Phase::VisibleIdle {
            return UxEvent::None;
        }
        if let Some(p) = self.press {
            if !self.dragging && (x - p.x).hypot(y - p.y) >= TAP_MAX_MOVE {
                self.dragging = true;
                return UxEvent::BeginDrag;
            }
        }
        UxEvent::None
    }

    pub fn button_up(&mut self, x: f32, y: f32, t_ms: u64, listen_seed: u64) -> UxEvent {
        if self.phase != Phase::VisibleIdle {
            self.press = None;
            self.dragging = false;
            return UxEvent::None;
        }
        let some = self.press.take();
        let was_drag = self.dragging;
        self.dragging = false;
        if let Some(p) = some {
            if was_drag {
                return UxEvent::EndDrag;
            }
            if is_tap(x - p.x, y - p.y, t_ms.saturating_sub(p.t_ms)) {
                self.enter_listen(t_ms, listen_seed);
                return UxEvent::StartListen;
            }
        }
        UxEvent::None
    }

    pub fn enter_listen(&mut self, now_ms: u64, seed: u64) {
        self.enter_listen_for(now_ms, listen_duration_ms(seed) as u64);
    }

    pub fn enter_listen_for(&mut self, now_ms: u64, duration_ms: u64) {
        self.phase = Phase::Listening;
        self.listen_until_ms = now_ms + duration_ms.max(1);
        self.press = None;
        self.dragging = false;
    }

    /// While listening, hide once `now_ms` reaches the deadline.
    pub fn tick(&mut self, now_ms: u64) -> UxEvent {
        if self.phase == Phase::Listening && now_ms >= self.listen_until_ms {
            self.phase = Phase::Hidden;
            return UxEvent::Hide;
        }
        UxEvent::None
    }

    /// Hidden → visible on any mouse movement or key (caller detects the input).
    pub fn wake(&mut self) -> UxEvent {
        if self.phase == Phase::Hidden {
            self.phase = Phase::VisibleIdle;
            self.press = None;
            self.dragging = false;
            return UxEvent::Wake;
        }
        UxEvent::None
    }
}

impl Default for BuddyUx {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_is_small_and_quick() {
        assert!(is_tap(0.0, 0.0, 80));
        assert!(is_tap(3.0, 4.0, 200)); // 5px
        assert!(!is_tap(7.0, 0.0, 80));
        assert!(!is_tap(0.0, 0.0, 250));
        assert!(!is_tap(10.0, 10.0, 400));
    }

    #[test]
    fn listen_duration_in_range() {
        for seed in [0u64, 1, 17, 999, 1_000_000] {
            let d = listen_duration_ms(seed);
            assert!((LISTEN_MIN_MS..=LISTEN_MAX_MS).contains(&d), "{d}");
        }
        assert_eq!(listen_duration_ms(0), LISTEN_MIN_MS);
        assert_eq!(listen_duration_ms(10_000), LISTEN_MAX_MS);
    }

    #[test]
    fn click_in_place_starts_listen() {
        let mut ux = BuddyUx::new();
        assert_eq!(ux.button_down(100.0, 100.0, 1000, true), UxEvent::None);
        assert_eq!(
            ux.button_up(101.0, 100.0, 1120, 3),
            UxEvent::StartListen
        );
        assert_eq!(ux.phase, Phase::Listening);
        assert!(ux.listen_until_ms > 1120);
    }

    #[test]
    fn drag_is_not_a_click() {
        let mut ux = BuddyUx::new();
        ux.button_down(100.0, 100.0, 1000, true);
        assert_eq!(ux.pointer_move(120.0, 100.0, 1050), UxEvent::BeginDrag);
        assert_eq!(ux.button_up(130.0, 110.0, 1180, 0), UxEvent::EndDrag);
        assert_eq!(ux.phase, Phase::VisibleIdle);
    }

    #[test]
    fn listen_then_hide_then_wake() {
        let mut ux = BuddyUx::new();
        ux.enter_listen(1_000, 0);
        assert_eq!(ux.phase, Phase::Listening);
        assert_eq!(ux.tick(2_000), UxEvent::None);
        assert_eq!(ux.tick(1_000 + LISTEN_MIN_MS as u64), UxEvent::Hide);
        assert_eq!(ux.phase, Phase::Hidden);
        assert_eq!(ux.wake(), UxEvent::Wake);
        assert_eq!(ux.phase, Phase::VisibleIdle);
    }

    #[test]
    fn click_outside_hit_does_nothing() {
        let mut ux = BuddyUx::new();
        assert_eq!(ux.button_down(0.0, 0.0, 0, false), UxEvent::None);
        assert_eq!(ux.button_up(0.0, 0.0, 10, 0), UxEvent::None);
        assert_eq!(ux.phase, Phase::VisibleIdle);
    }
}
