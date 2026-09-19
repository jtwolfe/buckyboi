//! Webcam capture for approximate gaze and identity frames. Feature `gaze`.
//! Opening the device is best-effort: failure returns `None` and the overlay
//! keeps mouse-avoid + click-to-listen.

use std::sync::mpsc::Receiver;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Copy, Debug)]
pub struct CamGaze {
    pub x: f32,
    pub y: f32,
    pub t_ms: u64,
}

#[derive(Clone, Debug)]
pub struct CamFrame {
    pub rgb: Vec<u8>,
    pub w: u32,
    pub h: u32,
    pub t_ms: u64,
}

static LATEST: OnceLock<Mutex<Option<CamFrame>>> = OnceLock::new();

fn latest_slot() -> &'static Mutex<Option<CamFrame>> {
    LATEST.get_or_init(|| Mutex::new(None))
}

pub fn latest_frame() -> Option<CamFrame> {
    latest_slot().lock().ok().and_then(|g| g.clone())
}

pub fn store_frame(frame: CamFrame) {
    if let Ok(mut g) = latest_slot().lock() {
        *g = Some(frame);
    }
}

/// Try to start a capture thread. `None` = no camera / permission / compile without `gaze`.
pub fn start(screen_w: f32, screen_h: f32) -> Option<Receiver<CamGaze>> {
    if crate::identity::env_flag_alias("BUCKYBOI_NO_CAMERA", "BUDDY_NO_CAMERA") {
        eprintln!("buckyboi: BUCKYBOI_NO_CAMERA set — skipping webcam");
        return None;
    }
    #[cfg(feature = "gaze")]
    {
        return start_v4l(screen_w, screen_h);
    }
    #[cfg(not(feature = "gaze"))]
    {
        let _ = (screen_w, screen_h);
        eprintln!("buckyboi: compiled without `gaze` feature — no webcam");
        None
    }
}

#[cfg(feature = "gaze")]
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(feature = "gaze")]
fn camera_path() -> String {
    crate::identity::env_or_alias("BUCKYBOI_CAMERA", "BUDDY_CAMERA")
        .unwrap_or_else(|| "/dev/video0".into())
}

#[cfg(feature = "gaze")]
fn start_v4l(screen_w: f32, screen_h: f32) -> Option<Receiver<CamGaze>> {
    use crate::gaze::{estimate_face_rgb, face_to_screen, yuyv_to_rgb};
    use std::sync::mpsc::{channel, sync_channel};
    use std::thread;
    use std::time::Duration;
    use v4l::buffer::Type;
    use v4l::io::traits::CaptureStream;
    use v4l::video::Capture;
    use v4l::{Device, FourCC};

    let path = camera_path();
    let (ready_tx, ready_rx) = channel::<Result<String, String>>();
    let (tx, rx) = sync_channel::<CamGaze>(1);

    if thread::Builder::new()
        .name("buckyboi-cam".into())
        .spawn(move || {
            let mut dev = match Device::with_path(&path) {
                Ok(d) => d,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{path}: {e}")));
                    return;
                }
            };
            let mut fmt = match dev.format() {
                Ok(f) => f,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("format: {e}")));
                    return;
                }
            };
            fmt.width = 320;
            fmt.height = 240;
            let yuyv = FourCC::new(b"YUYV");
            let mjpg = FourCC::new(b"MJPG");
            fmt.fourcc = yuyv;
            if dev.set_format(&fmt).is_err() {
                fmt.fourcc = mjpg;
                if let Err(e) = dev.set_format(&fmt) {
                    let _ = ready_tx.send(Err(format!("set_format: {e}")));
                    return;
                }
            }
            let fmt = match dev.format() {
                Ok(f) => f,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("format: {e}")));
                    return;
                }
            };
            let fourcc = fmt.fourcc;
            let fw = fmt.width;
            let fh = fmt.height;
            let mut stream =
                match v4l::io::mmap::Stream::with_buffers(&mut dev, Type::VideoCapture, 4) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("mmap: {e}")));
                        return;
                    }
                };
            let _ = ready_tx.send(Ok(format!(
                "{path} {}x{} {}",
                fw,
                fh,
                String::from_utf8_lossy(&fourcc.repr)
            )));

            let mut misses = 0u32;
            loop {
                let (buf, _meta) = match stream.next() {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("buckyboi: camera frame error ({e}) — stopping gaze");
                        break;
                    }
                };
                let rgb = if fourcc == yuyv {
                    yuyv_to_rgb(buf, fw, fh)
                } else if fourcc == mjpg {
                    decode_mjpeg(buf)
                } else if buf.len() >= (fw * fh * 3) as usize {
                    buf[..(fw * fh * 3) as usize].to_vec()
                } else {
                    continue;
                };
                if rgb.len() < (fw * fh * 3) as usize {
                    continue;
                }
                store_frame(CamFrame {
                    rgb: rgb.clone(),
                    w: fw,
                    h: fh,
                    t_ms: now_ms(),
                });
                let Some(guess) = estimate_face_rgb(&rgb, fw, fh) else {
                    misses = misses.saturating_add(1);
                    if misses == 45 {
                        eprintln!("buckyboi: camera open but no face yet (lighting / framing)");
                    }
                    continue;
                };
                misses = 0;
                let (lx, ly) = guess.look_point();
                let (sx, sy) =
                    face_to_screen(lx, ly, fw as f32, fh as f32, screen_w, screen_h, true);
                match tx.try_send(CamGaze {
                    x: sx,
                    y: sy,
                    t_ms: now_ms(),
                }) {
                    Ok(()) | Err(std::sync::mpsc::TrySendError::Full(_)) => {}
                    Err(std::sync::mpsc::TrySendError::Disconnected(_)) => break,
                }
            }
        })
        .is_err()
    {
        eprintln!("buckyboi: could not spawn camera thread — no gaze");
        return None;
    }

    match ready_rx.recv_timeout(Duration::from_secs(3)) {
        Ok(Ok(info)) => {
            eprintln!("buckyboi: camera {info} (face/iris proxy — not research-grade gaze)");
            Some(rx)
        }
        Ok(Err(e)) => {
            eprintln!("buckyboi: camera open failed ({e}) — mouse-avoid + click-to-listen");
            None
        }
        Err(_) => {
            eprintln!("buckyboi: camera timed out opening — mouse-avoid + click-to-listen");
            None
        }
    }
}

#[cfg(feature = "gaze")]
fn decode_mjpeg(bytes: &[u8]) -> Vec<u8> {
    match image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg) {
        Ok(img) => img.to_rgb8().into_raw(),
        Err(_) => Vec::new(),
    }
}
