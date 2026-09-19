//! Fullscreen X11 overlay buddy. Transparent, input-shaped, no ASR/LLM.
//! See README.md and UX.md.

use buckyboi::{
    camera, chase_gaze, click_listening_ex, corner_on, draw, gaze_over_hysteresis, hit_rects,
    hit_test, initial_on, load_settings, overlay_bounds, save_settings, spin,
    step_overlay_avoid, AuthSession, BuddyUx, EnrollKind, EnrollPhase, EnrollSession, GazeLock,
    GazeSmoother, GestureAction, GestureClass, IdentityHud, Phase, ProfileStore,
    RadialAction, RadialMenu, Settings, SettingsPage, UxEvent, HIT_RADIUS,
};
use buckyboi::identity::{env_flag, env_flag_alias, env_or_alias, extract_face, hands, voice};
use std::env;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use x11rb::connection::{Connection, RequestConnection};
use x11rb::protocol::shape::{self, ConnectionExt as _};
use x11rb::protocol::xfixes::ConnectionExt as _;
use x11rb::protocol::xproto::*;
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;

const ESCAPE_KEYSYM: u32 = 0xFF1B;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn listen_seed() -> u64 {
    now_ms()
}

fn listen_override_ms() -> Option<u64> {
    env_or_alias("BUCKYBOI_LISTEN_MS", "BUDDY_LISTEN_MS")
        .and_then(|v| v.parse::<u64>().ok())
        .map(|ms| ms.max(1))
}

fn gaze_lock_override_ms() -> Option<u32> {
    env_or_alias("BUCKYBOI_GAZE_LOCK_MS", "BUDDY_GAZE_LOCK_MS")
        .and_then(|v| v.parse::<u32>().ok())
        .map(|ms| ms.max(1))
}

fn hud_from(
    auth: &AuthSession,
    profiles: &ProfileStore,
    _settings: &Settings,
    enroll: &EnrollSession,
    now: u64,
) -> IdentityHud {
    IdentityHud {
        auth_label: auth.state.label(now),
        people: profiles.people().iter().map(|p| p.name.clone()).collect(),
        enroll_hint: enroll.hint(),
        enroll_progress: enroll.progress(),
        enroll_active: !matches!(enroll.phase, EnrollPhase::Idle),
    }
}

fn finish_enroll(enroll: &mut EnrollSession, profiles: &mut ProfileStore, settings: &mut Settings) {
    if !matches!(enroll.phase, EnrollPhase::Done { .. }) {
        return;
    }
    let name = if enroll.person_name.trim().is_empty() {
        profiles.next_default_name()
    } else {
        enroll.person_name.clone()
    };
    let id = enroll
        .person_id
        .clone()
        .unwrap_or_else(|| profiles.upsert_named(&name));
    if enroll.person_id.is_none() {
        enroll.person_id = Some(id.clone());
    }
    match enroll.kind {
        EnrollKind::Face => {
            profiles.replace_face(&id, enroll.accepted.clone());
            eprintln!("buckyboi: enrolled face for {name} ({} vectors)", enroll.accepted.len());
        }
        EnrollKind::Voice => {
            profiles.replace_voice(&id, enroll.accepted.clone());
            eprintln!("buckyboi: enrolled voice for {name} ({} vectors)", enroll.accepted.len());
        }
        EnrollKind::Gesture => {
            if let Some(cls) = enroll.gesture {
                profiles.add_gesture_samples(&id, cls, enroll.gesture_samples.clone());
                eprintln!(
                    "buckyboi: calibrated {} for {name} ({} samples)",
                    cls.label(),
                    enroll.gesture_samples.len()
                );
            }
        }
    }
    profiles.save();
    if let Some(i) = profiles.people().iter().position(|p| p.id == id) {
        settings.selected_person = i;
    }
    settings.page = SettingsPage::People;
    *enroll = EnrollSession::idle();
}

fn synthetic_skin() -> (Vec<u8>, u32, u32) {
    let w = 80u32;
    let h = 60u32;
    let mut rgb = vec![18u8; (w * h * 3) as usize];
    for y in 10..48 {
        for x in 16..56 {
            let i = ((y * w + x) * 3) as usize;
            let tex = if (x + y) % 3 == 0 { 70 } else { 188 };
            rgb[i] = tex;
            rgb[i + 1] = 118;
            rgb[i + 2] = 98;
        }
    }
    (rgb, w, h)
}

fn identity_tick(
    now: u64,
    profiles: &mut ProfileStore,
    auth: &mut AuthSession,
    enroll: &mut EnrollSession,
    settings: &Settings,
    last_face_ms: &mut u64,
    last_voice_ms: &mut u64,
    last_hand_ms: &mut u64,
) {
    auth.expire(now);

    let frame = if env_flag("BUCKYBOI_FACE_SIM") {
        let (rgb, w, h) = synthetic_skin();
        Some(camera::CamFrame {
            rgb,
            w,
            h,
            t_ms: now,
        })
    } else {
        camera::latest_frame()
    };

    if let Some(frame) = frame {
        if now.saturating_sub(*last_face_ms) >= 180 {
            *last_face_ms = now;
            let (q, emb) = if matches!(enroll.kind, EnrollKind::Face)
                && !matches!(enroll.phase, EnrollPhase::Idle)
            {
                extract_face(&frame.rgb, frame.w, frame.h)
            } else {
                extract_face(&frame.rgb, frame.w, frame.h)
            };
            if matches!(enroll.kind, EnrollKind::Face)
                && !matches!(enroll.phase, EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. })
            {
                let ev = enroll.push_face(q, emb.clone());
                if ev == buckyboi::identity::EnrollEvent::Finished {
                    // finished below
                }
            } else if let Some(emb) = emb {
                if let Some(hit) = profiles.match_face(&emb) {
                    auth.note_face(&hit.person_id, &hit.name, now, hit.score);
                }
            }
        }
    }

    if matches!(enroll.kind, EnrollKind::Voice)
        && !matches!(enroll.phase, EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. })
        && now.saturating_sub(*last_voice_ms) >= 1_600
    {
        *last_voice_ms = now;
        let samples = voice_samples();
        if !samples.is_empty() {
            let (q, emb) = voice::extract_embedding(&samples, voice::VOICE_SAMPLE_RATE);
            let _ = enroll.push_voice(q, emb);
        }
    } else if matches!(enroll.phase, EnrollPhase::Idle) && now.saturating_sub(*last_voice_ms) >= 2_400 {
        if let Some(samples) = voice_samples_optional() {
            *last_voice_ms = now;
            let (q, emb) = voice::extract_embedding(&samples, voice::VOICE_SAMPLE_RATE);
            if q.ok {
                if let Some(emb) = emb {
                    if let Some(hit) = profiles.match_voice(&emb) {
                        auth.note_voice(&hit.person_id, &hit.name, now, hit.score);
                    }
                }
            }
        }
    }

    if now.saturating_sub(*last_hand_ms) >= 220 {
        if let Some(hand) = current_hand() {
            *last_hand_ms = now;
            if matches!(enroll.kind, EnrollKind::Gesture)
                && !matches!(enroll.phase, EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. })
            {
                let _ = enroll.push_gesture(&hand.normalized());
            }
        }
    }

    if matches!(enroll.phase, EnrollPhase::Done { .. }) {
        // caller persists
    }
    let _ = settings;
}

fn voice_tone() -> Vec<f32> {
    let n = voice::VOICE_SAMPLE_RATE * 1600 / 1000;
    (0..n)
        .map(|i| 0.22 * (2.0 * std::f32::consts::PI * 196.0 * i as f32 / voice::VOICE_SAMPLE_RATE as f32).sin())
        .collect()
}

fn voice_samples() -> Vec<f32> {
    if env_flag("BUCKYBOI_VOICE_SIM") {
        return voice_tone();
    }
    #[cfg(feature = "voice")]
    {
        // Live mic is started lazily from main when voice enroll begins.
    }
    Vec::new()
}

fn voice_samples_optional() -> Option<Vec<f32>> {
    if env_flag("BUCKYBOI_VOICE_SIM") {
        return Some(voice_tone());
    }
    None
}

fn current_hand() -> Option<hands::HandLandmarks> {
    if let Some(sim) = env::var("BUCKYBOI_HAND_SIM").ok() {
        let cls = GestureClass::parse(&sim);
        if cls != GestureClass::Unknown {
            return Some(hands::synthetic(cls));
        }
    }
    None
}

fn apply_identity_action(
    act: RadialAction,
    settings: &mut Settings,
    profiles: &mut ProfileStore,
    enroll: &mut EnrollSession,
    auth: &mut AuthSession,
) {
    match act {
        RadialAction::CycleGate | RadialAction::ToggleNeedFace | RadialAction::TabLook | RadialAction::TabPeople => {
            save_settings(settings);
        }
        RadialAction::NewPerson => {
            let name = profiles.next_default_name();
            let id = profiles.upsert_named(&name);
            profiles.save();
            if let Some(i) = profiles.people().iter().position(|p| p.id == id) {
                settings.selected_person = i;
            }
            settings.page = SettingsPage::People;
            eprintln!("buckyboi: new person {name}");
        }
        RadialAction::DeleteSelected => {
            if let Some(p) = profiles.people().get(settings.selected_person).cloned() {
                profiles.delete(&p.id);
                profiles.save();
                settings.selected_person = 0;
                *auth = AuthSession::new(8_000);
                eprintln!("buckyboi: deleted {}", p.name);
            }
        }
        RadialAction::StartEnrollFace => {
            let (name, id) = person_target(profiles, settings);
            *enroll = EnrollSession::start_face(name, id);
            enroll.begin_capture();
            eprintln!("buckyboi: face enroll — look at the camera (or BUCKYBOI_FACE_SIM=1)");
        }
        RadialAction::StartEnrollVoice => {
            let (name, id) = person_target(profiles, settings);
            *enroll = EnrollSession::start_voice(name, id);
            enroll.begin_capture();
            eprintln!("buckyboi: voice enroll — speak for ~1.5s × 3 (or BUCKYBOI_VOICE_SIM=1)");
        }
        RadialAction::StartEnrollHands => {
            let (name, id) = person_target(profiles, settings);
            *enroll = EnrollSession::start_gesture(name, id, GestureClass::Fist, hands::GESTURE_ENROLL_NEED);
            enroll.begin_capture();
            eprintln!("buckyboi: gesture calibrate — hold FIST (BUCKYBOI_HAND_SIM=fist …)");
        }
        RadialAction::CancelEnroll => {
            enroll.cancel();
            eprintln!("buckyboi: enroll cancelled");
        }
        _ => {}
    }
}

fn person_target(profiles: &mut ProfileStore, settings: &Settings) -> (String, Option<String>) {
    if let Some(p) = profiles.people().get(settings.selected_person) {
        return (p.name.clone(), Some(p.id.clone()));
    }
    let name = profiles.next_default_name();
    let id = profiles.upsert_named(&name);
    profiles.save();
    (name, Some(id))
}

fn maybe_gesture_action(
    now: u64,
    profiles: &ProfileStore,
    auth: &AuthSession,
    settings: &Settings,
    last_hand_ms: &mut u64,
) -> Option<GestureAction> {
    if now.saturating_sub(*last_hand_ms) < 480 {
        return None;
    }
    let hand = current_hand()?;
    if !auth.allows_gesture(settings.gate, settings.gestures_need_face, now) {
        return None;
    }
    let cents = auth
        .state
        .person_id
        .as_ref()
        .and_then(|id| profiles.get(id))
        .map(|p| p.gesture_centroids())
        .unwrap_or_default();
    let (cls, score) = hands::classify_gesture(&hand, &cents, profiles.file.gesture_threshold);
    if cls == GestureClass::Unknown || score < 0.7 {
        return None;
    }
    let map = auth
        .state
        .person_id
        .as_ref()
        .and_then(|id| profiles.get(id))
        .map(|p| p.map())
        .unwrap_or(hands::DEFAULT_GESTURE_MAP);
    *last_hand_ms = now;
    Some(hands::action_for(cls, &map))
}

enum GazeDrive {
    Off,
    Camera(std::sync::mpsc::Receiver<camera::CamGaze>),
    Mouse,
    Chase { origin_ms: u64 },
}

fn find_argb_visual(screen: &Screen) -> Option<(u8, Visualid)> {
    for depth in &screen.allowed_depths {
        if depth.depth != 32 {
            continue;
        }
        for vis in &depth.visuals {
            if vis.class == VisualClass::TRUE_COLOR {
                return Some((32, vis.visual_id));
            }
        }
    }
    None
}

fn intern(conn: &RustConnection, name: &[u8]) -> Atom {
    conn.intern_atom(false, name)
        .ok()
        .and_then(|c| c.reply().ok())
        .map(|r| r.atom)
        .unwrap_or(0)
}

fn set_dock_hints(conn: &RustConnection, win: Window) -> Result<(), Box<dyn std::error::Error>> {
    let t = intern(conn, b"_NET_WM_WINDOW_TYPE");
    let dock = intern(conn, b"_NET_WM_WINDOW_TYPE_DOCK");
    conn.change_property32(PropMode::REPLACE, win, t, AtomEnum::ATOM, &[dock])?;

    let state = intern(conn, b"_NET_WM_STATE");
    let above = intern(conn, b"_NET_WM_STATE_ABOVE");
    let sticky = intern(conn, b"_NET_WM_STATE_STICKY");
    let skip_task = intern(conn, b"_NET_WM_STATE_SKIP_TASKBAR");
    let skip_pager = intern(conn, b"_NET_WM_STATE_SKIP_PAGER");
    conn.change_property32(
        PropMode::REPLACE,
        win,
        state,
        AtomEnum::ATOM,
        &[above, sticky, skip_task, skip_pager],
    )?;

    // Do not reserve a dock strut.
    let strut = intern(conn, b"_NET_WM_STRUT_PARTIAL");
    conn.change_property32(PropMode::REPLACE, win, strut, AtomEnum::CARDINAL, &[0; 12])?;

    conn.change_property8(
        PropMode::REPLACE,
        win,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        b"buckyboi",
    )?;
    Ok(())
}

fn escape_keycodes(conn: &RustConnection, setup: &Setup) -> Vec<u8> {
    let min = setup.min_keycode;
    let count = setup.max_keycode.saturating_sub(min).saturating_add(1);
    let Ok(cookie) = conn.get_keyboard_mapping(min, count) else {
        return vec![9];
    };
    let Ok(reply) = cookie.reply() else {
        return vec![9];
    };
    let w = reply.keysyms_per_keycode as usize;
    let mut out = Vec::new();
    for (i, chunk) in reply.keysyms.chunks(w).enumerate() {
        if chunk.iter().any(|k| *k == ESCAPE_KEYSYM) {
            out.push(min + i as u8);
        }
    }
    if out.is_empty() {
        out.push(9);
    }
    out
}

fn keymap_has(keys: &[u8], code: u8) -> bool {
    let i = code as usize;
    i < keys.len() * 8 && keys[i / 8] & (1 << (i % 8)) != 0
}

fn any_escape(keys: &[u8], codes: &[u8]) -> bool {
    codes.iter().any(|c| keymap_has(keys, *c))
}

/// Input (and bounding) region. Empty = fully click-through.
fn set_hit_region(
    conn: &RustConnection,
    win: Window,
    hits: &[(i16, i16, u16, u16)],
) -> Result<(), Box<dyn std::error::Error>> {
    let rects: Vec<Rectangle> = hits
        .iter()
        .map(|&(x, y, width, height)| Rectangle {
            x,
            y,
            width,
            height,
        })
        .collect();
    conn.shape_rectangles(
        shape::SO::SET,
        shape::SK::INPUT,
        ClipOrdering::UNSORTED,
        win,
        0,
        0,
        &rects,
    )?;
    conn.shape_rectangles(
        shape::SO::SET,
        shape::SK::BOUNDING,
        ClipOrdering::UNSORTED,
        win,
        0,
        0,
        &rects,
    )?;
    let region = conn.generate_id()?;
    conn.xfixes_create_region(region, &rects)?;
    conn.xfixes_set_window_shape_region(win, shape::SK::INPUT, 0, 0, region)?;
    conn.xfixes_set_window_shape_region(win, shape::SK::BOUNDING, 0, 0, region)?;
    conn.xfixes_destroy_region(region)?;
    Ok(())
}

fn bounds_rect(b: (f32, f32, f32, f32), sw: u16, sh: u16) -> (i16, i16, u16, u16) {
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

fn clear_rect(
    conn: &RustConnection,
    win: Window,
    gc: Gcontext,
    x: i16,
    y: i16,
    w: u16,
    h: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = w as usize * h as usize * 4;
    let zeros = vec![0u8; n];
    conn.put_image(
        ImageFormat::Z_PIXMAP,
        win,
        gc,
        w,
        h,
        x,
        y,
        0,
        32,
        &zeros,
    )?;
    Ok(())
}

fn paint_buddy(
    conn: &RustConnection,
    win: Window,
    gc: Gcontext,
    state: &buckyboi::State,
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
) -> Result<(), Box<dyn std::error::Error>> {
    let (ox, oy, w, h) = origin;
    let mut buf = vec![0u8; w as usize * h as usize * 4];
    let off_x = ox as f32;
    let off_y = oy as f32;
    draw::paint_wireframe(
        &mut buf,
        w,
        h,
        state,
        off_x,
        off_y,
        listening,
        settings.stroke,
    );
    if !listening && dwell > 0.04 {
        let rad = HIT_RADIUS * (0.62 + 0.28 * dwell);
        let alpha = (36.0 + 90.0 * dwell) as u8;
        draw::stroke(
            &mut buf,
            w,
            h,
            state.cx - off_x + rad,
            state.cy - off_y,
            state.cx - off_x + rad,
            state.cy - off_y,
            1.0,
            0xC8,
            0xD0,
            0xDC,
            alpha,
        );
        const N: i32 = 64;
        for i in 0..N {
            let t0 = i as f32 * (std::f32::consts::TAU / N as f32);
            let t1 = (i + 1) as f32 * (std::f32::consts::TAU / N as f32);
            draw::stroke(
                &mut buf,
                w,
                h,
                state.cx - off_x + rad * t0.cos(),
                state.cy - off_y + rad * t0.sin(),
                state.cx - off_x + rad * t1.cos(),
                state.cy - off_y + rad * t1.sin(),
                1.6,
                0xC8,
                0xD0,
                0xDC,
                alpha,
            );
        }
    }
    if listening {
        let rad = HIT_RADIUS * (0.72 + 0.10 * pulse);
        let alpha = (50.0 + 90.0 * pulse) as u8;
        const N: i32 = 64;
        for i in 0..N {
            let t0 = i as f32 * (std::f32::consts::TAU / N as f32);
            let t1 = (i + 1) as f32 * (std::f32::consts::TAU / N as f32);
            draw::stroke(
                &mut buf,
                w,
                h,
                state.cx - off_x + rad * t0.cos(),
                state.cy - off_y + rad * t0.sin(),
                state.cx - off_x + rad * t1.cos(),
                state.cy - off_y + rad * t1.sin(),
                1.8,
                0x7E,
                0xE8,
                0xFF,
                alpha,
            );
        }
        draw::paint_listening_chrome(
            &mut buf,
            w,
            h,
            state.cx,
            state.cy,
            menu,
            settings,
            hud,
            mx,
            my,
            now_ms,
            screen_w,
            screen_h,
            off_x,
            off_y,
        );
    }
    let known = hud.auth_label != "UNKNOWN" && !hud.auth_label.is_empty();
    if !hud.auth_label.is_empty() {
        draw::paint_auth_chip(
            &mut buf,
            w,
            h,
            state.cx,
            state.cy,
            &hud.auth_label,
            known,
            off_x,
            off_y,
        );
    }
    conn.put_image(
        ImageFormat::Z_PIXMAP,
        win,
        gc,
        w,
        h,
        ox,
        oy,
        0,
        32,
        &buf,
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::args().any(|a| a == "-h" || a == "--help") {
        println!(
            "buckyboi — Linux overlay buddy (gaze / voice / gesture identity)\n\n\
             Usage: buckyboi\n\n\
             Fullscreen X11 overlay. Look at the icosahedron (or click it) to listen\n\
             if the gate allows. Settings → ID enrolls face, voice, and gestures for\n\
             one or many people. Profiles stay in ~/.config/buckyboi/ (offline).\n\
             Esc quits. Native Wayland is not supported.\n\n\
             Features (cargo): gaze (default), face, hands, voice, voice-sherpa\n\n\
             Env:\n\
               DISPLAY                 X11 display (required)\n\
               BUCKYBOI_LISTEN_MS      override listen duration (ms)\n\
               BUCKYBOI_GAZE_LOCK_MS   override gaze-follow duration (ms)\n\
               BUCKYBOI_CAMERA         V4L2 device (default /dev/video0)\n\
               BUCKYBOI_NO_CAMERA      skip webcam\n\
               BUCKYBOI_GAZE_SIM       mouse | chase\n\
               BUCKYBOI_CONFIG         override config dir (tests)\n\
               BUCKYBOI_MODELS         ONNX model directory\n\
               BUCKYBOI_FACE_PROBE     weak local face print if no ArcFace model\n\
               BUCKYBOI_FACE_SIM       synthetic face frames (no camera)\n\
               BUCKYBOI_VOICE_SIM      tone | file  (no mic)\n\
               BUCKYBOI_HAND_SIM       fist|palm|thumb|point|peace\n\n\
             BUDDY_* aliases still work. See README.md."
        );
        return Ok(());
    }

    if env::var_os("DISPLAY").is_none() {
        if env::var_os("WAYLAND_DISPLAY").is_some() {
            eprintln!(
                "buckyboi: native Wayland has no portable global-input wake.\n\
                 Start an XWayland session (DISPLAY=:0) or see README.md."
            );
            std::process::exit(1);
        }
        eprintln!("buckyboi: DISPLAY is unset; need X11. See README.md.");
        std::process::exit(1);
    }

    let (conn, screen_num) = RustConnection::connect(None)?;
    let setup = conn.setup().clone();
    let screen = &setup.roots[screen_num];
    let sw = screen.width_in_pixels;
    let sh = screen.height_in_pixels;
    let root = screen.root;

    conn.extension_information(shape::X11_EXTENSION_NAME)?
        .ok_or("X Shape extension missing")?;
    conn.xfixes_query_version(5, 0)?.reply()?;

    let (depth, visual) = find_argb_visual(screen).ok_or("no 32-bit ARGB visual")?;
    let cmap = conn.generate_id()?;
    conn.create_colormap(ColormapAlloc::NONE, cmap, root, visual)?;

    let win = conn.generate_id()?;
    let aux = CreateWindowAux::new()
        .event_mask(
            EventMask::EXPOSURE
                | EventMask::STRUCTURE_NOTIFY
                | EventMask::BUTTON_PRESS
                | EventMask::BUTTON_RELEASE
                | EventMask::POINTER_MOTION,
        )
        .override_redirect(1)
        .colormap(cmap)
        .border_pixel(0)
        .background_pixel(0);
    conn.create_window(
        depth,
        win,
        root,
        0,
        0,
        sw,
        sh,
        0,
        WindowClass::INPUT_OUTPUT,
        visual,
        &aux,
    )?;
    set_dock_hints(&conn, win)?;
    // Never intercept the desktop until the buddy AABB is known.
    set_hit_region(&conn, win, &[])?;

    let gc = conn.generate_id()?;
    conn.create_gc(gc, win, &CreateGCAux::new())?;

    conn.map_window(win)?;
    conn.configure_window(
        win,
        &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE),
    )?;
    conn.flush()?;

    let esc = escape_keycodes(&conn, &setup);
    let mut state = initial_on(sw as f32, sh as f32);
    let (home_x, home_y) = corner_on(0, state.sw, state.sh);
    state.cx = home_x;
    state.cy = home_y;
    state.mode = buckyboi::LOCKED;
    let mut ux = BuddyUx::new();
    let mut menu = RadialMenu::new();
    let mut settings = load_settings();
    if env_flag_alias("BUCKYBOI_NO_CAMERA", "BUDDY_NO_CAMERA") {
        settings.camera = false;
    }
    if let Some(ms) = listen_override_ms() {
        settings.listen_ms = ms as u32;
    }
    if let Some(ms) = gaze_lock_override_ms() {
        settings.gaze_lock_ms = ms;
    }
    let mut gaze_smooth = GazeSmoother::new();
    let mut gaze_lock = match gaze_lock_override_ms() {
        Some(ms) => GazeLock::with_need(ms),
        None => GazeLock::with_need(settings.gaze_lock_ms),
    };
    let mut profiles = ProfileStore::load();
    profiles.file.face_threshold = profiles.file.face_threshold.max(0.15);
    let mut auth = AuthSession::new(8_000);
    let mut enroll = EnrollSession::idle();
    let mut last_hand_ms = 0u64;
    let mut last_voice_ms = 0u64;
    let mut last_face_ms = 0u64;

    let mut gaze_drive = match env_or_alias("BUCKYBOI_GAZE_SIM", "BUDDY_GAZE_SIM")
        .as_deref()
    {
        Some("mouse") => {
            eprintln!("buckyboi: GAZE_SIM=mouse — pointer is a stand-in gaze");
            GazeDrive::Mouse
        }
        Some("chase") => {
            eprintln!("buckyboi: GAZE_SIM=chase — scripted look-at-buddy demo");
            GazeDrive::Chase {
                origin_ms: now_ms(),
            }
        }
        Some(other) => {
            eprintln!("buckyboi: unknown GAZE_SIM={other} (use mouse|chase)");
            GazeDrive::Off
        }
        None if settings.camera => match camera::start(sw as f32, sh as f32) {
            Some(rx) => GazeDrive::Camera(rx),
            None => GazeDrive::Off,
        },
        None => {
            eprintln!("buckyboi: camera disabled in settings — mouse-avoid + click-to-listen");
            GazeDrive::Off
        }
    };

    let mut prev_button = false;
    let mut last_root = (0i16, 0i16);
    let mut have_last_root = false;
    let mut last_keys = [0u8; 32];
    let mut have_keys = false;
    let mut prev_rect: Option<(i16, i16, u16, u16)> = None;
    let mut mapped = true;
    let pad = HIT_RADIUS + 28.0;
    let mut last_raise = Instant::now();

    eprintln!(
        "buckyboi overlay {}x{} — look or click to listen (if gate allows), Esc quits",
        sw, sh
    );
    if let Some(ms) = listen_override_ms() {
        eprintln!("BUDDY_LISTEN_MS override → {ms} ms");
    }
    if let Some(ms) = gaze_lock_override_ms() {
        eprintln!("BUDDY_GAZE_LOCK_MS override → {ms} ms");
    } else {
        eprintln!(
            "buckyboi: gaze-follow lock {} ms (random 5–15s unless overridden)",
            gaze_lock.need_ms
        );
    }

    let frame = Duration::from_micros(16_666);
    loop {
        let tick_start = Instant::now();
        let mut ev_press = false;
        let mut ev_release = false;
        let mut ev_xy: Option<(f32, f32)> = None;
        while let Some(ev) = conn.poll_for_event()? {
            match ev {
                Event::Error(e) => eprintln!("X error: {e:?}"),
                Event::ButtonPress(e) if e.detail == 1 => {
                    ev_press = true;
                    ev_xy = Some((e.root_x as f32, e.root_y as f32));
                }
                Event::ButtonRelease(e) if e.detail == 1 => {
                    ev_release = true;
                    ev_xy = Some((e.root_x as f32, e.root_y as f32));
                }
                Event::MotionNotify(e) => {
                    ev_xy = Some((e.root_x as f32, e.root_y as f32));
                }
                _ => {}
            }
        }

        let ptr = conn.query_pointer(root)?.reply()?;
        let mx = ev_xy.map(|(x, _)| x).unwrap_or(ptr.root_x as f32);
        let my = ev_xy.map(|(_, y)| y).unwrap_or(ptr.root_y as f32);
        // QueryPointer can miss a <16 ms click; ButtonPress/Release on the
        // shaped hit region cannot. OR them so short taps still count.
        let button = ptr.mask.contains(KeyButMask::BUTTON1) || (ev_press && !ev_release);
        let keys = conn.query_keymap()?.reply()?.keys;

        if any_escape(&keys, &esc) {
            break;
        }

        let now = now_ms();
        identity_tick(
            now,
            &mut profiles,
            &mut auth,
            &mut enroll,
            &settings,
            &mut last_face_ms,
            &mut last_voice_ms,
            &mut last_hand_ms,
        );
        if matches!(enroll.phase, EnrollPhase::Done { .. }) {
            finish_enroll(&mut enroll, &mut profiles, &mut settings);
        }
        if matches!(enroll.phase, EnrollPhase::Idle) {
        if let Some(gact) = maybe_gesture_action(now, &profiles, &auth, &settings, &mut last_hand_ms)
        {
            match gact {
                GestureAction::Listen if ux.phase == Phase::VisibleIdle => {
                    if auth.allows_listen(settings.gate, now) {
                        let listen_ms = listen_override_ms().unwrap_or(settings.listen_ms as u64);
                        ux.enter_listen_for(now, listen_ms);
                        menu.reset();
                        gaze_lock.reset();
                        eprintln!("buckyboi: listening… (gesture palm)");
                    }
                }
                GestureAction::Dismiss => {
                    ux.phase = Phase::Hidden;
                    eprintln!("buckyboi: dismiss (gesture)");
                }
                GestureAction::Settings if ux.phase == Phase::Listening => {
                    menu.settings_open = !menu.settings_open;
                }
                GestureAction::Mute => {
                    menu.muted = !menu.muted;
                    eprintln!("buckyboi: mute → {}", menu.muted);
                }
                GestureAction::Info => {
                    eprintln!(
                        "buckyboi: {} via {}",
                        auth.state.label(now),
                        settings.gate.label()
                    );
                }
                _ => {}
            }
        }
        }
        let mut gaze_pt: Option<(f32, f32)> = None;
        match &gaze_drive {
            GazeDrive::Off => {}
            GazeDrive::Camera(rx) => {
                while let Ok(s) = rx.try_recv() {
                    gaze_smooth.push(s.x, s.y, s.t_ms);
                }
                gaze_pt = gaze_smooth.current(now);
            }
            GazeDrive::Mouse => {
                gaze_pt = Some(gaze_smooth.push(mx, my, now));
            }
            GazeDrive::Chase { origin_ms } => {
                let (gx, gy) = chase_gaze(now, *origin_ms, state.cx, state.cy, sw as f32, sh as f32);
                gaze_pt = Some(gaze_smooth.push(gx, gy, now));
            }
        }

        match ux.phase {
            Phase::Hidden => {
                let moved = have_last_root
                    && ((ptr.root_x - last_root.0).abs() >= 1
                        || (ptr.root_y - last_root.1).abs() >= 1);
                let keyed = have_keys && keys != last_keys;
                if moved || keyed {
                    let _ = ux.wake();
                    menu.reset();
                    gaze_lock = match gaze_lock_override_ms() {
                        Some(ms) => GazeLock::with_need(ms),
                        None => GazeLock::with_need(settings.gaze_lock_ms),
                    };
                    gaze_smooth.clear();
                    if let GazeDrive::Chase { origin_ms } = &mut gaze_drive {
                        *origin_ms = now;
                    }
                    eprintln!("buckyboi: wake");
                    conn.map_window(win)?;
                    conn.configure_window(
                        win,
                        &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE),
                    )?;
                    mapped = true;
                }
            }
            Phase::VisibleIdle => {
                let over = hit_test(&state, mx, my);
                if (button && !prev_button) || ev_press {
                    let _ = ux.button_down(mx, my, now, over);
                }
                if button || ev_press || ux.press.is_some() {
                    if ux.pointer_move(mx, my, now) == UxEvent::BeginDrag {
                        eprintln!("buckyboi: drag");
                        gaze_lock.reset();
                    }
                }
                if ev_release || (!button && prev_button) {
                    match ux.button_up(mx, my, now, listen_seed()) {
                        UxEvent::StartListen => {
                            if !auth.allows_listen(settings.gate, now) {
                                eprintln!(
                                    "buckyboi: listen blocked (gate {} / {})",
                                    settings.gate.label(),
                                    auth.state.label(now)
                                );
                            } else {
                                if let Some(ms) = listen_override_ms() {
                                    ux.enter_listen_for(now, ms);
                                } else {
                                    ux.enter_listen_for(now, settings.listen_ms as u64);
                                }
                                menu.reset();
                                gaze_lock.reset();
                                eprintln!(
                                    "buckyboi: listening… ({} ms, click, {})",
                                    ux.listen_until_ms.saturating_sub(now),
                                    auth.state.label(now)
                                );
                            }
                        }
                        UxEvent::EndDrag => {
                            eprintln!("buckyboi: drag release → new corner");
                        }
                        _ => {}
                    }
                }
                if ux.phase == Phase::VisibleIdle {
                    if let Some((gx, gy)) = gaze_pt {
                        let dist = (gx - state.cx).hypot(gy - state.cy);
                        let over = gaze_over_hysteresis(gaze_lock.on, dist, HIT_RADIUS);
                        if gaze_lock.update(now, over) == UxEvent::StartListen {
                            if !auth.allows_listen(settings.gate, now) {
                                gaze_lock.reset();
                                eprintln!(
                                    "buckyboi: gaze-lock ignored (gate {} / {})",
                                    settings.gate.label(),
                                    auth.state.label(now)
                                );
                            } else {
                                let listen_ms = listen_override_ms()
                                    .unwrap_or(settings.listen_ms as u64);
                                ux.enter_listen_for(now, listen_ms);
                                menu.reset();
                                gaze_lock.reset();
                                eprintln!(
                                    "buckyboi: listening… ({} ms, gaze lock, {})",
                                    ux.listen_until_ms.saturating_sub(now),
                                    auth.state.label(now)
                                );
                            }
                        }
                    } else {
                        gaze_lock.update(now, false);
                    }
                }
                if ux.phase == Phase::VisibleIdle {
                    step_overlay_avoid(&mut state, mx, my, ux.dragging, gaze_pt);
                } else {
                    spin(&mut state);
                }
            }
            Phase::Listening => {
                spin(&mut state);
                if ev_release || (!button && prev_button) {
                    let act = click_listening_ex(
                        &mut menu,
                        &mut settings,
                        mx,
                        my,
                        state.cx,
                        state.cy,
                        now,
                        sw as f32,
                        sh as f32,
                        profiles.enrolled_count(),
                        !matches!(enroll.phase, EnrollPhase::Idle),
                    );
                    apply_identity_action(
                        act,
                        &mut settings,
                        &mut profiles,
                        &mut enroll,
                        &mut auth,
                    );
                    match act {
                        RadialAction::Dismiss => {
                            eprintln!("buckyboi: dismiss");
                            ux.phase = Phase::Hidden;
                        }
                        RadialAction::Mute => {
                            eprintln!(
                                "buckyboi: mute stub → {}",
                                if menu.muted { "on" } else { "off" }
                            );
                        }
                        RadialAction::Info => {
                            eprintln!(
                                "buckyboi: {} people, gate {}, auth {}",
                                profiles.enrolled_count(),
                                settings.gate.label(),
                                auth.state.label(now)
                            );
                        }
                        RadialAction::ToggleSettings => {
                            eprintln!(
                                "buckyboi: settings {}",
                                if menu.settings_open { "open" } else { "closed" }
                            );
                        }
                        RadialAction::CloseSettings => {
                            eprintln!("buckyboi: settings closed");
                        }
                        RadialAction::ToggleCamera => {
                            eprintln!(
                                "buckyboi: camera setting → {}",
                                if settings.camera { "on" } else { "off" }
                            );
                            save_settings(&settings);
                            if settings.camera {
                                if !matches!(gaze_drive, GazeDrive::Mouse | GazeDrive::Chase { .. }) {
                                    gaze_drive = match camera::start(sw as f32, sh as f32) {
                                        Some(rx) => GazeDrive::Camera(rx),
                                        None => GazeDrive::Off,
                                    };
                                }
                            } else if matches!(gaze_drive, GazeDrive::Camera(_)) {
                                gaze_drive = GazeDrive::Off;
                            }
                        }
                        RadialAction::SetListenMs(_)
                        | RadialAction::SetGazeMs(_)
                        | RadialAction::SetStroke(_) => {
                            save_settings(&settings);
                            if gaze_lock_override_ms().is_none() {
                                gaze_lock.need_ms = settings.gaze_lock_ms;
                            }
                        }
                        RadialAction::StartEnrollFace
                        | RadialAction::StartEnrollVoice
                        | RadialAction::StartEnrollHands => {
                            settings.page = SettingsPage::People;
                            menu.settings_open = true;
                        }
                        RadialAction::None
                        | RadialAction::CycleGate
                        | RadialAction::ToggleNeedFace
                        | RadialAction::TabLook
                        | RadialAction::TabPeople
                        | RadialAction::NewPerson
                        | RadialAction::SelectPerson(_)
                        | RadialAction::DeleteSelected
                        | RadialAction::CancelEnroll => {}
                    }
                }
                if !menu.settings_open && ux.tick(now) == UxEvent::Hide {
                    ux.phase = Phase::Hidden;
                }
                if ux.phase == Phase::Hidden {
                    menu.reset();
                    eprintln!("buckyboi: hidden (move the mouse or press a key to return)");
                    if let Some((x, y, w, h)) = prev_rect.take() {
                        let _ = clear_rect(&conn, win, gc, x, y, w, h);
                    }
                    let _ = set_hit_region(&conn, win, &[]);
                    conn.unmap_window(win)?;
                    mapped = false;
                }
            }
        }

        if mapped && ux.phase != Phase::Hidden {
            let listening = ux.phase == Phase::Listening;
            let bounds = overlay_bounds(
                state.cx,
                state.cy,
                pad,
                listening,
                &menu,
                now,
                sw as f32,
                sh as f32,
            );
            let rect = bounds_rect(bounds, sw, sh);
            if let Some(old) = prev_rect {
                if old != rect {
                    clear_rect(&conn, win, gc, old.0, old.1, old.2, old.3)?;
                }
            }
            let pulse = if listening {
                0.5 + 0.5 * ((now as f32 / 280.0).sin())
            } else {
                0.0
            };
            let dwell = if ux.phase == Phase::VisibleIdle {
                gaze_lock.progress(now)
            } else {
                0.0
            };
            let hud = hud_from(&auth, &profiles, &settings, &enroll, now);
            paint_buddy(
                &conn,
                win,
                gc,
                &state,
                listening,
                pulse,
                dwell,
                &menu,
                &settings,
                &hud,
                mx,
                my,
                now,
                sw as f32,
                sh as f32,
                rect,
            )?;
            let hits = hit_rects(
                state.cx,
                state.cy,
                pad,
                listening,
                &menu,
                now,
                sw as f32,
                sh as f32,
            );
            set_hit_region(&conn, win, &hits)?;
            prev_rect = Some(rect);
            if last_raise.elapsed() > Duration::from_secs(2) {
                conn.configure_window(
                    win,
                    &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE),
                )?;
                last_raise = Instant::now();
            }
        }

        last_root = (ptr.root_x, ptr.root_y);
        have_last_root = true;
        last_keys = keys;
        have_keys = true;
        prev_button = button;
        conn.flush()?;

        let spent = tick_start.elapsed();
        if let Some(rest) = frame.checked_sub(spent) {
            // Drain without blocking input for long.
            std::thread::sleep(rest.min(Duration::from_millis(8)));
        }
    }

    let _ = set_hit_region(&conn, win, &[]);
    conn.unmap_window(win)?;
    conn.destroy_window(win)?;
    conn.flush()?;
    eprintln!("buckyboi: quit");
    Ok(())
}
