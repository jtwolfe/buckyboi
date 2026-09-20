//! Wake / quit channel for the running overlay.
//!
//! Wayland cannot portably watch “any key / mouse anywhere”. On Omarchy /
//! Hyprland we:
//!   1. Poll `cursorpos` over Hyprland IPC (mouse move while hidden).
//!   2. Listen on `$XDG_RUNTIME_DIR/buckyboi.sock` for `wake` / `quit`
//!      (`buckyboi --wake`, `buckyboi --quit`, or a Hyprland bind).
//!   3. Honor SIGUSR1 (wake) and SIGINT/SIGTERM (quit).
//!
//! True “any key anywhere” is not possible without a compositor bind.

use serde::Deserialize;
use std::fs;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static WANT_WAKE: AtomicBool = AtomicBool::new(false);
static WANT_QUIT: AtomicBool = AtomicBool::new(false);
static SIGNALS_ON: AtomicBool = AtomicBool::new(false);

extern "C" fn on_usr1(_: libc::c_int) {
    WANT_WAKE.store(true, Ordering::SeqCst);
}

extern "C" fn on_quit(_: libc::c_int) {
    WANT_QUIT.store(true, Ordering::SeqCst);
}

pub fn install_signals() {
    if SIGNALS_ON.swap(true, Ordering::SeqCst) {
        return;
    }
    unsafe {
        libc::signal(libc::SIGUSR1, on_usr1 as usize);
        libc::signal(libc::SIGINT, on_quit as usize);
        libc::signal(libc::SIGTERM, on_quit as usize);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WakeCommand {
    Wake,
    Quit,
}

pub fn parse_command(line: &str) -> Option<WakeCommand> {
    match line.trim().to_ascii_lowercase().as_str() {
        "wake" | "show" | "unhide" => Some(WakeCommand::Wake),
        "quit" | "exit" | "die" => Some(WakeCommand::Quit),
        _ => None,
    }
}

pub fn socket_path() -> PathBuf {
    if let Ok(p) = std::env::var("BUCKYBOI_SOCK") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let runtime = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(runtime).join("buckyboi.sock")
}

pub fn send_command(cmd: WakeCommand) -> Result<(), String> {
    let path = socket_path();
    let mut stream = UnixStream::connect(&path).map_err(|e| {
        format!(
            "buckyboi: no running overlay at {} ({e}). Start `buckyboi` first.",
            path.display()
        )
    })?;
    stream
        .set_write_timeout(Some(Duration::from_millis(400)))
        .ok();
    let word = match cmd {
        WakeCommand::Wake => "wake\n",
        WakeCommand::Quit => "quit\n",
    };
    stream
        .write_all(word.as_bytes())
        .map_err(|e| format!("buckyboi: failed to send {word:?}: {e}"))?;
    Ok(())
}

pub fn parse_cursorpos(s: &str) -> Option<(i32, i32)> {
    let s = s.trim();
    let (x, y) = s.split_once(',')?;
    Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
}

#[derive(Debug, Deserialize)]
struct HyprMonitor {
    #[serde(default)]
    name: String,
    #[serde(default)]
    x: i32,
    #[serde(default)]
    y: i32,
    #[serde(default)]
    width: i32,
    #[serde(default)]
    height: i32,
    #[serde(default)]
    focused: bool,
}

#[derive(Clone, Debug)]
pub struct HyprOutput {
    pub name: String,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

pub fn hypr_socket_dir() -> Option<PathBuf> {
    let sig = std::env::var("HYPRLAND_INSTANCE_SIGNATURE").ok()?;
    if sig.is_empty() {
        return None;
    }
    let runtime = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let dir = PathBuf::from(runtime).join("hypr").join(sig);
    dir.is_dir().then_some(dir)
}

pub fn hypr_available() -> bool {
    hypr_socket_dir()
        .map(|d| d.join(".socket.sock").exists())
        .unwrap_or(false)
}

pub fn hypr_command(cmd: &str) -> Option<String> {
    let dir = hypr_socket_dir()?;
    let mut stream = UnixStream::connect(dir.join(".socket.sock")).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_millis(50)))
        .ok()?;
    stream
        .set_write_timeout(Some(Duration::from_millis(50)))
        .ok()?;
    stream.write_all(cmd.as_bytes()).ok()?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).ok()?;
    Some(buf)
}

pub fn hypr_cursorpos() -> Option<(i32, i32)> {
    parse_cursorpos(&hypr_command("cursorpos")?)
}

pub fn hypr_focused_monitor() -> Option<HyprOutput> {
    let raw = hypr_command("j/monitors")?;
    let list: Vec<HyprMonitor> = serde_json::from_str(&raw).ok()?;
    let m = list.iter().find(|m| m.focused).or_else(|| list.first())?;
    Some(HyprOutput {
        name: m.name.clone(),
        x: m.x,
        y: m.y,
        width: m.width,
        height: m.height,
    })
}

/// Convert Hyprland global cursor to overlay-local pixels.
pub fn hypr_local_cursor(mon: &HyprOutput, gx: i32, gy: i32) -> (f32, f32) {
    ((gx - mon.x) as f32, (gy - mon.y) as f32)
}

pub struct WakeBus {
    listener: Option<UnixListener>,
    path: PathBuf,
    last_cursor: Option<(i32, i32)>,
    pub hypr: Option<HyprOutput>,
}

impl WakeBus {
    pub fn bind() -> Self {
        install_signals();
        let path = socket_path();
        let listener = bind_socket(&path);
        if listener.is_some() {
            eprintln!("buckyboi: control socket {}", path.display());
        }
        let hypr = hypr_focused_monitor();
        if let Some(ref m) = hypr {
            eprintln!(
                "buckyboi: Hyprland IPC on {} ({}x{}+{}+{}) — cursor wake enabled",
                m.name, m.width, m.height, m.x, m.y
            );
        } else if std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some() {
            eprintln!("buckyboi: HYPRLAND_INSTANCE_SIGNATURE set but IPC socket missing");
        }
        Self {
            listener,
            path,
            last_cursor: None,
            hypr,
        }
    }

    pub fn refresh_hypr_monitor(&mut self) {
        if let Some(m) = hypr_focused_monitor() {
            self.hypr = Some(m);
        }
    }

    /// Drain socket + signals + (when `watch_cursor`) Hyprland cursor motion.
    pub fn drain(&mut self, watch_cursor: bool) -> (bool, bool, Option<(f32, f32)>) {
        let mut wake = WANT_WAKE.swap(false, Ordering::SeqCst);
        let mut quit = WANT_QUIT.swap(false, Ordering::SeqCst);
        if let Some(ref listener) = self.listener {
            loop {
                match listener.accept() {
                    Ok((stream, _)) => match read_command(stream) {
                        Some(WakeCommand::Wake) => wake = true,
                        Some(WakeCommand::Quit) => quit = true,
                        None => {}
                    },
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(_) => break,
                }
            }
        }
        let mut local_cursor = None;
        if watch_cursor {
            if let Some((gx, gy)) = hypr_cursorpos() {
                if let Some(prev) = self.last_cursor {
                    if (gx - prev.0).abs() >= 1 || (gy - prev.1).abs() >= 1 {
                        wake = true;
                    }
                }
                self.last_cursor = Some((gx, gy));
                if let Some(ref mon) = self.hypr {
                    local_cursor = Some(hypr_local_cursor(mon, gx, gy));
                } else {
                    local_cursor = Some((gx as f32, gy as f32));
                }
            }
        } else if let Some((gx, gy)) = hypr_cursorpos() {
            self.last_cursor = Some((gx, gy));
            if let Some(ref mon) = self.hypr {
                local_cursor = Some(hypr_local_cursor(mon, gx, gy));
            } else {
                local_cursor = Some((gx as f32, gy as f32));
            }
        }
        (wake, quit, local_cursor)
    }
}

impl Drop for WakeBus {
    fn drop(&mut self) {
        self.listener.take();
        let _ = fs::remove_file(&self.path);
    }
}

fn bind_socket(path: &Path) -> Option<UnixListener> {
    if path.exists() {
        if UnixStream::connect(path).is_ok() {
            eprintln!(
                "buckyboi: another instance holds {} — this process will not own the socket",
                path.display()
            );
            return None;
        }
        let _ = fs::remove_file(path);
    }
    match UnixListener::bind(path) {
        Ok(l) => {
            let _ = l.set_nonblocking(true);
            Some(l)
        }
        Err(e) => {
            eprintln!("buckyboi: cannot bind {}: {e}", path.display());
            None
        }
    }
}

fn read_command(mut stream: UnixStream) -> Option<WakeCommand> {
    stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .ok()?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf).ok()?;
    parse_command(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_wake_words() {
        assert_eq!(parse_command("wake\n"), Some(WakeCommand::Wake));
        assert_eq!(parse_command("SHOW"), Some(WakeCommand::Wake));
        assert_eq!(parse_command("quit"), Some(WakeCommand::Quit));
        assert_eq!(parse_command("nope"), None);
    }

    #[test]
    fn parse_hypr_cursorpos() {
        assert_eq!(parse_cursorpos("1920,540\n"), Some((1920, 540)));
        assert_eq!(parse_cursorpos(" 10 , 20 "), Some((10, 20)));
        assert_eq!(parse_cursorpos("oops"), None);
    }

    #[test]
    fn local_cursor_subtracts_monitor_origin() {
        let mon = HyprOutput {
            name: "DP-2".into(),
            x: 1920,
            y: 0,
            width: 1440,
            height: 900,
        };
        assert_eq!(hypr_local_cursor(&mon, 2000, 100), (80.0, 100.0));
    }

    #[test]
    fn hypr_monitors_json() {
        let raw = r#"[{"name":"eDP-1","x":0,"y":0,"width":1920,"height":1080,"focused":true}]"#;
        let list: Vec<HyprMonitor> = serde_json::from_str(raw).unwrap();
        assert_eq!(list[0].width, 1920);
        assert!(list[0].focused);
    }

    #[test]
    fn control_socket_wake_roundtrip() {
        let dir = std::env::temp_dir().join(format!("buckyboi-sock-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("buckyboi.sock");
        std::env::set_var("BUCKYBOI_SOCK", &path);
        let mut bus = WakeBus::bind();
        send_command(WakeCommand::Wake).expect("send wake");
        // Accept is non-blocking; give the listener a tick.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let (wake, quit, _) = bus.drain(false);
        assert!(wake, "wake command should land");
        assert!(!quit);
        drop(bus);
        let _ = std::fs::remove_dir_all(dir);
        std::env::remove_var("BUCKYBOI_SOCK");
    }
}
