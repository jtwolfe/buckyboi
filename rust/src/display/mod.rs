//! Display backends: native Wayland (wlr-layer-shell) and X11 Shape/XFixes.
//!
//! Runtime pick: `WAYLAND_DISPLAY` + a working layer-shell compositor wins;
//! otherwise `DISPLAY` / X11. Override with `BUCKYBOI_DISPLAY=wayland|x11`.

pub mod wake;
pub mod wayland;
pub mod x11;

use crate::draw;
use crate::radial::{IdentityHud, RadialMenu, Settings};
use crate::sim::State;

pub use wake::{send_command, socket_path, WakeBus, WakeCommand};
pub use wayland::WaylandDisplay;
pub use x11::X11Display;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Wayland,
    X11,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectError {
    NoneAvailable,
    WaylandForcedMissing,
    X11ForcedMissing,
    UnknownForce,
}

/// Pointer / key snapshot for one overlay frame.
#[derive(Clone, Debug, Default)]
pub struct FrameInput {
    pub mx: f32,
    pub my: f32,
    pub button: bool,
    pub ev_press: bool,
    pub ev_release: bool,
    pub ev_xy: Option<(f32, f32)>,
    pub escape: bool,
    pub wake: bool,
    pub quit: bool,
}

pub enum Backend {
    Wayland(WaylandDisplay),
    X11(X11Display),
}

impl Backend {
    pub fn kind(&self) -> BackendKind {
        match self {
            Backend::Wayland(_) => BackendKind::Wayland,
            Backend::X11(_) => BackendKind::X11,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Backend::Wayland(_) => "wayland",
            Backend::X11(_) => "x11",
        }
    }

    pub fn size(&self) -> (u16, u16) {
        match self {
            Backend::Wayland(d) => d.size(),
            Backend::X11(d) => d.size(),
        }
    }

    pub fn poll_input(&mut self) -> Result<FrameInput, Box<dyn std::error::Error>> {
        match self {
            Backend::Wayland(d) => d.poll_input(),
            Backend::X11(d) => d.poll_input(),
        }
    }

    pub fn present(
        &mut self,
        pixels: &[u8],
        origin: (i16, i16, u16, u16),
        hits: &[(i16, i16, u16, u16)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Backend::Wayland(d) => d.present(pixels, origin, hits),
            Backend::X11(d) => d.present(pixels, origin, hits),
        }
    }

    pub fn hide(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Backend::Wayland(d) => d.hide(),
            Backend::X11(d) => d.hide(),
        }
    }

    pub fn show(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Backend::Wayland(d) => d.show(),
            Backend::X11(d) => d.show(),
        }
    }

    pub fn raise(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Backend::Wayland(d) => d.raise(),
            Backend::X11(d) => d.raise(),
        }
    }

    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Backend::Wayland(d) => d.flush(),
            Backend::X11(d) => d.flush(),
        }
    }

    pub fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Backend::Wayland(d) => d.shutdown(),
            Backend::X11(d) => d.shutdown(),
        }
    }
}

pub fn force_from_env() -> Option<String> {
    std::env::var("BUCKYBOI_DISPLAY")
        .ok()
        .or_else(|| std::env::var("BUDDY_DISPLAY").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Choose a backend kind from environment presence + optional force flag.
pub fn select_backend_kind(
    wayland_display: bool,
    x11_display: bool,
    force: Option<&str>,
) -> Result<BackendKind, SelectError> {
    match force.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if s == "wayland" || s == "wl" || s == "native" => {
            if wayland_display {
                Ok(BackendKind::Wayland)
            } else {
                Err(SelectError::WaylandForcedMissing)
            }
        }
        Some(s) if s == "x11" || s == "x" => {
            if x11_display {
                Ok(BackendKind::X11)
            } else {
                Err(SelectError::X11ForcedMissing)
            }
        }
        Some(s) if s == "auto" || s == "default" => {
            select_backend_kind(wayland_display, x11_display, None)
        }
        Some(_) => Err(SelectError::UnknownForce),
        None => {
            if wayland_display {
                Ok(BackendKind::Wayland)
            } else if x11_display {
                Ok(BackendKind::X11)
            } else {
                Err(SelectError::NoneAvailable)
            }
        }
    }
}

pub fn open_backend() -> Result<Backend, Box<dyn std::error::Error>> {
    let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
    let x11 = std::env::var_os("DISPLAY").is_some();
    let force = force_from_env();
    let kind = select_backend_kind(wayland, x11, force.as_deref()).map_err(|e| match e {
        SelectError::NoneAvailable => {
            "buckyboi: neither WAYLAND_DISPLAY nor DISPLAY is set. See README.md.".to_string()
        }
        SelectError::WaylandForcedMissing => {
            "buckyboi: BUCKYBOI_DISPLAY=wayland but WAYLAND_DISPLAY is unset.".to_string()
        }
        SelectError::X11ForcedMissing => {
            "buckyboi: BUCKYBOI_DISPLAY=x11 but DISPLAY is unset.".to_string()
        }
        SelectError::UnknownForce => {
            "buckyboi: BUCKYBOI_DISPLAY must be wayland, x11, or auto.".to_string()
        }
    })?;

    match kind {
        BackendKind::Wayland => match WaylandDisplay::open() {
            Ok(d) => {
                eprintln!(
                    "buckyboi: Wayland wlr-layer-shell overlay {}x{}",
                    d.size().0,
                    d.size().1
                );
                Ok(Backend::Wayland(d))
            }
            Err(e) => {
                if x11 && force.as_deref().is_none() {
                    eprintln!("buckyboi: Wayland layer-shell failed ({e}); falling back to X11");
                    Ok(Backend::X11(X11Display::open()?))
                } else {
                    Err(e)
                }
            }
        },
        BackendKind::X11 => Ok(Backend::X11(X11Display::open()?)),
    }
}

pub fn bounds_rect(b: (f32, f32, f32, f32), sw: u16, sh: u16) -> (i16, i16, u16, u16) {
    let (x, y, w, h) = b;
    let x0 = x.floor().max(0.0) as i32;
    let y0 = y.floor().max(0.0) as i32;
    let x1 = (x + w).ceil().min(sw as f32) as i32;
    let y1 = (y + h).ceil().min(sh as f32) as i32;
    (
        x0 as i16,
        y0 as i16,
        (x1 - x0).max(1) as u16,
        (y1 - y0).max(1) as u16,
    )
}

pub fn paint_rect(
    state: &State,
    listening: bool,
    pulse: f32,
    dwell: f32,
    menu: &RadialMenu,
    settings: &Settings,
    hud: &IdentityHud,
    mx: f32,
    my: f32,
    now_ms: u64,
    screen_w: f32,
    screen_h: f32,
    origin: (i16, i16, u16, u16),
) -> Vec<u8> {
    let (ox, oy, w, h) = origin;
    let mut buf = vec![0u8; w as usize * h as usize * 4];
    draw::paint_overlay(
        &mut buf, w, h, state, listening, pulse, dwell, menu, settings, hud, mx, my, now_ms,
        screen_w, screen_h, ox as f32, oy as f32,
    );
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_prefers_wayland_when_both_set() {
        assert_eq!(
            select_backend_kind(true, true, None).unwrap(),
            BackendKind::Wayland
        );
    }

    #[test]
    fn auto_uses_x11_without_wayland() {
        assert_eq!(
            select_backend_kind(false, true, None).unwrap(),
            BackendKind::X11
        );
    }

    #[test]
    fn force_x11_even_if_wayland_present() {
        assert_eq!(
            select_backend_kind(true, true, Some("x11")).unwrap(),
            BackendKind::X11
        );
    }

    #[test]
    fn force_wayland_without_socket_fails() {
        assert_eq!(
            select_backend_kind(false, true, Some("wayland")),
            Err(SelectError::WaylandForcedMissing)
        );
    }

    #[test]
    fn none_available() {
        assert_eq!(
            select_backend_kind(false, false, None),
            Err(SelectError::NoneAvailable)
        );
    }

    #[test]
    fn bounds_rect_clamps() {
        let r = bounds_rect((-10.0, -4.0, 40.0, 20.0), 100, 80);
        assert_eq!(r, (0, 0, 30, 16));
    }

    #[test]
    fn both_backends_are_linked() {
        let _ = BackendKind::Wayland;
        let _ = BackendKind::X11;
        let _ = crate::display::wayland::BTN_LEFT;
        let _ = crate::display::x11::X11_BACKEND_NAME;
    }
}
