//! Webcam capture for approximate gaze and identity frames. Feature `gaze`.
//! Opening the device is best-effort: failure returns `None` and the overlay
//! keeps mouse-avoid + click-to-listen.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Copy, Debug)]
pub struct CamGaze {
    pub x: f32,
    pub y: f32,
    pub t_ms: u64,
}

#[derive(Clone, Debug)]
pub struct CamFrame {
    pub rgb: Arc<[u8]>,
    pub w: u32,
    pub h: u32,
    pub t_ms: u64,
}

/// Generation captured at spawn; the loop breaks if this no longer matches.
static CAM_EPOCH: AtomicU64 = AtomicU64::new(0);

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

pub fn clear_frame() {
    if let Ok(mut g) = latest_slot().lock() {
        *g = None;
    }
}

/// Invalidate the running capture thread and drop the RGB slot.
pub fn request_stop() {
    CAM_EPOCH.fetch_add(1, Ordering::SeqCst);
    clear_frame();
}

/// Try to start a capture thread. `None` = no camera / permission / compile without `gaze`.
pub fn start(screen_w: f32, screen_h: f32) -> Option<Receiver<CamGaze>> {
    if crate::identity::env_flag_alias("BUCKYBOI_NO_CAMERA", "BUDDY_NO_CAMERA") {
        eprintln!("buckyboi: BUCKYBOI_NO_CAMERA set — skipping webcam");
        request_stop();
        return None;
    }
    let my_epoch = CAM_EPOCH.fetch_add(1, Ordering::SeqCst) + 1;
    #[cfg(feature = "gaze")]
    {
        start_v4l(screen_w, screen_h, my_epoch)
    }
    #[cfg(not(feature = "gaze"))]
    {
        let _ = (screen_w, screen_h, my_epoch);
        clear_frame();
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
fn env_dim(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|&n| (16..=4096).contains(&n))
}

#[cfg(feature = "gaze")]
fn negotiate_format(dev: &mut v4l::Device) -> Result<v4l::Format, String> {
    use v4l::video::Capture;
    use v4l::FourCC;

    let yuyv = FourCC::new(b"YUYV");
    let mjpg = FourCC::new(b"MJPG");
    let mut candidates: Vec<(u32, u32, FourCC)> = Vec::new();
    if let (Some(w), Some(h)) = (env_dim("BUCKYBOI_CAMERA_W"), env_dim("BUCKYBOI_CAMERA_H")) {
        candidates.push((w, h, yuyv));
        candidates.push((w, h, mjpg));
    }
    candidates.extend_from_slice(&[
        (640, 480, yuyv),
        (640, 480, mjpg),
        (320, 240, yuyv),
        (320, 240, mjpg),
    ]);

    let mut last_err = String::from("no format");
    for (w, h, fcc) in candidates {
        let mut fmt = match dev.format() {
            Ok(f) => f,
            Err(e) => return Err(format!("format: {e}")),
        };
        fmt.width = w;
        fmt.height = h;
        fmt.fourcc = fcc;
        match dev.set_format(&fmt) {
            Ok(_) => {
                return dev.format().map_err(|e| format!("format: {e}"));
            }
            Err(e) => last_err = format!("set_format {w}x{h}: {e}"),
        }
    }
    Err(last_err)
}

#[cfg(feature = "gaze")]
fn start_v4l(screen_w: f32, screen_h: f32, my_epoch: u64) -> Option<Receiver<CamGaze>> {
    use crate::gaze::{estimate_face_rgb, face_to_screen, yuyv_to_rgb};
    use std::sync::mpsc::{channel, sync_channel};
    use std::thread;
    use std::time::Duration;
    use v4l::buffer::Type;
    use v4l::io::traits::CaptureStream;
    use v4l::{Device, FourCC};

    let path = camera_path();
    let (ready_tx, ready_rx) = channel::<Result<String, String>>();
    let (tx, rx) = sync_channel::<CamGaze>(1);

    if thread::Builder::new()
        .name("buckyboi-cam".into())
        .spawn(move || {
            struct ClearIfCurrent(u64);
            impl Drop for ClearIfCurrent {
                fn drop(&mut self) {
                    if CAM_EPOCH.load(Ordering::SeqCst) == self.0 {
                        clear_frame();
                    }
                }
            }
            let _clear = ClearIfCurrent(my_epoch);
            let mut dev = match Device::with_path(&path) {
                Ok(d) => d,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{path}: {e}")));
                    return;
                }
            };
            let fmt = match negotiate_format(&mut dev) {
                Ok(f) => f,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let fourcc = fmt.fourcc;
            let fw = fmt.width;
            let fh = fmt.height;
            let yuyv = FourCC::new(b"YUYV");
            let mjpg = FourCC::new(b"MJPG");
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
                if CAM_EPOCH.load(Ordering::SeqCst) != my_epoch {
                    break;
                }
                let (buf, _meta) = match stream.next() {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("buckyboi: camera frame error ({e}) — stopping gaze");
                        break;
                    }
                };
                if CAM_EPOCH.load(Ordering::SeqCst) != my_epoch {
                    break;
                }
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
                let rgb: Arc<[u8]> = Arc::from(rgb);
                if CAM_EPOCH.load(Ordering::SeqCst) != my_epoch {
                    break;
                }
                store_frame(CamFrame {
                    rgb: Arc::clone(&rgb),
                    w: fw,
                    h: fh,
                    t_ms: now_ms(),
                });
                #[cfg(feature = "face")]
                if crate::identity::publishing_gaze() {
                    continue;
                }
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
                    Ok(())
                    | Err(std::sync::mpsc::TrySendError::Full(_))
                    | Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {}
                }
            }
        })
        .is_err()
    {
        eprintln!("buckyboi: could not spawn camera thread — no gaze");
        if CAM_EPOCH.load(Ordering::SeqCst) == my_epoch {
            clear_frame();
        }
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
            request_stop();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camframe_arc_clone_shares_buffer() {
        let rgb: Arc<[u8]> = Arc::from(vec![9u8; 12]);
        let a = CamFrame {
            rgb: Arc::clone(&rgb),
            w: 2,
            h: 2,
            t_ms: 7,
        };
        let b = a.clone();
        assert!(Arc::ptr_eq(&a.rgb, &b.rgb));
        assert_eq!(a.w, 2);
        store_frame(a);
        let got = latest_frame().expect("stored");
        assert_eq!(got.t_ms, 7);
        assert!(Arc::ptr_eq(&got.rgb, &rgb));
    }

    #[test]
    fn request_stop_bumps_epoch_and_clears_slot() {
        let rgb: Arc<[u8]> = Arc::from(vec![1u8; 12]);
        store_frame(CamFrame {
            rgb,
            w: 2,
            h: 2,
            t_ms: 99,
        });
        let before = CAM_EPOCH.load(Ordering::SeqCst);
        request_stop();
        assert!(CAM_EPOCH.load(Ordering::SeqCst) > before);
        assert!(latest_frame().is_none());
    }
}
