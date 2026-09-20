//! Fullscreen overlay buddy. Wayland (wlr-layer-shell) or X11 Shape/XFixes.
//! See README.md and UX.md.

use buckyboi::display::{
    bounds_rect, open_backend, paint_rect, send_command, BackendKind, FrameInput, WakeBus,
    WakeCommand,
};
use buckyboi::identity::{
    env_flag, env_flag_alias, env_or_alias, extract_face, hands, latest_look, voice, VISION_INFER_MS,
};
use buckyboi::face_to_screen;
use buckyboi::{
    camera, chase_gaze, click_listening_ex, corner_on, gaze_over_hysteresis, hit_rects, hit_test,
    initial_on, load_settings, overlay_bounds, save_settings, spin, step_overlay_avoid,
    AuthSession, BuddyUx, EnrollKind, EnrollPhase, EnrollSession, GazeLock, GazeSmoother,
    GestureAction, GestureClass, IdentityHud, Phase, ProfileStore, RadialAction, RadialMenu,
    Settings, SettingsPage, UxEvent, HIT_RADIUS,
};
use std::env;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
            eprintln!(
                "buckyboi: enrolled face for {name} ({} vectors)",
                enroll.accepted.len()
            );
        }
        EnrollKind::Voice => {
            profiles.replace_voice(&id, enroll.accepted.clone());
            eprintln!(
                "buckyboi: enrolled voice for {name} ({} vectors)",
                enroll.accepted.len()
            );
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
        if now.saturating_sub(*last_face_ms) >= VISION_INFER_MS {
            *last_face_ms = now;
            let (_q, emb) = extract_face(&frame.rgb, frame.w, frame.h);
            if matches!(enroll.kind, EnrollKind::Face)
                && !matches!(
                    enroll.phase,
                    EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. }
                )
            {
                let ev = enroll.push_face(_q, emb.clone());
                let _ = ev;
            } else if let Some(emb) = emb {
                if let Some(hit) = profiles.match_face(&emb) {
                    auth.note_face(&hit.person_id, &hit.name, now, hit.score);
                }
            }
        }
    }

    if matches!(enroll.kind, EnrollKind::Voice)
        && !matches!(
            enroll.phase,
            EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. }
        )
        && now.saturating_sub(*last_voice_ms) >= 1_600
    {
        *last_voice_ms = now;
        let samples = voice_samples();
        if !samples.is_empty() {
            let (q, emb) = voice::extract_embedding(&samples, voice::VOICE_SAMPLE_RATE);
            let _ = enroll.push_voice(q, emb);
        }
    } else if matches!(enroll.phase, EnrollPhase::Idle)
        && now.saturating_sub(*last_voice_ms) >= 2_400
    {
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

    if now.saturating_sub(*last_hand_ms) >= VISION_INFER_MS {
        if let Some(hand) = current_hand() {
            *last_hand_ms = now;
            if matches!(enroll.kind, EnrollKind::Gesture)
                && !matches!(
                    enroll.phase,
                    EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. }
                )
            {
                let _ = enroll.push_gesture(&hand.normalized());
            }
        }
    }

    let _ = settings;
}

fn voice_tone() -> Vec<f32> {
    let n = voice::VOICE_SAMPLE_RATE * 1600 / 1000;
    (0..n)
        .map(|i| {
            0.22 * (2.0 * std::f32::consts::PI * 196.0 * i as f32 / voice::VOICE_SAMPLE_RATE as f32)
                .sin()
        })
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
    let frame = camera::latest_frame()?;
    hands::extract_landmarks_onnx(&frame.rgb, frame.w, frame.h)
}

fn apply_identity_action(
    act: RadialAction,
    settings: &mut Settings,
    profiles: &mut ProfileStore,
    enroll: &mut EnrollSession,
    auth: &mut AuthSession,
) {
    match act {
        RadialAction::CycleGate
        | RadialAction::ToggleNeedFace
        | RadialAction::TabLook
        | RadialAction::TabPeople => {
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
            *enroll = EnrollSession::start_gesture(
                name,
                id,
                GestureClass::Fist,
                hands::GESTURE_ENROLL_NEED,
            );
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

fn print_help() {
    println!(
        "buckyboi — Linux overlay buddy (gaze / voice / gesture identity)\n\n\
         Usage: buckyboi [--wake|--quit]\n\n\
         Fullscreen overlay on Wayland (wlr-layer-shell / Hyprland / Omarchy)\n\
         or X11 (Shape + XFixes). Look at the icosahedron (or click it) to\n\
         listen if the gate allows. Settings → ID enrolls face, voice, and\n\
         gestures. Profiles stay in ~/.config/buckyboi/ (offline).\n\n\
         Commands:\n\
           buckyboi            run the overlay\n\
           buckyboi --wake     wake a hidden instance (Hyprland bind)\n\
           buckyboi --quit     quit a running instance\n\n\
         Esc quits when the overlay has keyboard focus (X11: keymap poll).\n\
         On Wayland, also bind SUPER+Escape → `buckyboi --quit`.\n\n\
         Features (cargo): gaze (default), face, hands, voice\n\
         (`voice` includes sherpa-onnx; `voice-sherpa` is an alias)\n\n\
         Env:\n\
           WAYLAND_DISPLAY         prefer native Wayland layer-shell\n\
           DISPLAY                 X11 / XWayland fallback\n\
           BUCKYBOI_DISPLAY        wayland | x11 | auto\n\
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
           BUCKYBOI_HAND_SIM       fist|palm|thumb|point|peace\n\
           BUCKYBOI_SOCK           control socket (default $XDG_RUNTIME_DIR/buckyboi.sock)\n\n\
         BUDDY_* aliases still work. See README.md and contrib/omarchy/."
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return Ok(());
    }
    if args.iter().any(|a| a == "--wake" || a == "wake") {
        send_command(WakeCommand::Wake).map_err(|e| e)?;
        eprintln!("buckyboi: wake sent");
        return Ok(());
    }
    if args.iter().any(|a| a == "--quit" || a == "quit") {
        send_command(WakeCommand::Quit).map_err(|e| e)?;
        eprintln!("buckyboi: quit sent");
        return Ok(());
    }

    let mut backend = open_backend()?;
    let mut wake = WakeBus::bind();
    let (mut sw, mut sh) = backend.size();

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

    let mut gaze_drive = match env_or_alias("BUCKYBOI_GAZE_SIM", "BUDDY_GAZE_SIM").as_deref() {
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
    let mut mapped = true;
    let pad = HIT_RADIUS + 28.0;

    eprintln!(
        "buckyboi overlay {}x{} ({}) — look or click to listen (if gate allows), Esc quits",
        sw,
        sh,
        backend.name()
    );
    if backend.kind() == BackendKind::Wayland {
        eprintln!(
            "buckyboi: Wayland wake = Hyprland cursor IPC and/or SUPER+B → `buckyboi --wake`.\n\
             True “any key anywhere” is not portable; see contrib/omarchy/hyprland.conf"
        );
    }
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
    let mut last_hypr_refresh = Instant::now();
    loop {
        let tick_start = Instant::now();
        let (nw, nh) = backend.size();
        if nw != sw || nh != sh {
            sw = nw;
            sh = nh;
            state.sw = sw as f32;
            state.sh = sh as f32;
        }

        let mut input: FrameInput = backend.poll_input()?;
        let hidden = ux.phase == Phase::Hidden;
        if last_hypr_refresh.elapsed() > Duration::from_secs(5) {
            wake.refresh_hypr_monitor();
            last_hypr_refresh = Instant::now();
        }
        let (wake_now, quit_now, hypr_xy) =
            wake.drain(hidden || backend.kind() == BackendKind::Wayland);
        if quit_now {
            input.quit = true;
        }
        if wake_now {
            input.wake = true;
        }
        if let Some((hx, hy)) = hypr_xy {
            // Global cursor (Hyprland) for mouse-avoid + hidden wake.
            if !input.ev_press && !input.ev_release {
                input.mx = hx;
                input.my = hy;
            }
        }
        if input.quit {
            break;
        }

        let mx = input.ev_xy.map(|(x, _)| x).unwrap_or(input.mx);
        let my = input.ev_xy.map(|(_, y)| y).unwrap_or(input.my);
        let button = input.button;
        let ev_press = input.ev_press;
        let ev_release = input.ev_release;

        if input.escape {
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
            if let Some(gact) =
                maybe_gesture_action(now, &profiles, &auth, &settings, &mut last_hand_ms)
            {
                match gact {
                    GestureAction::Listen if ux.phase == Phase::VisibleIdle => {
                        if auth.allows_listen(settings.gate, now) {
                            let listen_ms =
                                listen_override_ms().unwrap_or(settings.listen_ms as u64);
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
                if let Some(look) = latest_look(now, 220) {
                    let (sx, sy) = face_to_screen(
                        look.cx,
                        look.cy,
                        look.fw,
                        look.fh,
                        sw as f32,
                        sh as f32,
                        true,
                    );
                    gaze_pt = Some(gaze_smooth.push(sx, sy, now));
                } else {
                    gaze_pt = gaze_smooth.current(now);
                }
            }
            GazeDrive::Mouse => {
                gaze_pt = Some(gaze_smooth.push(mx, my, now));
            }
            GazeDrive::Chase { origin_ms } => {
                let (gx, gy) =
                    chase_gaze(now, *origin_ms, state.cx, state.cy, sw as f32, sh as f32);
                gaze_pt = Some(gaze_smooth.push(gx, gy, now));
            }
        }

        match ux.phase {
            Phase::Hidden => {
                if input.wake {
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
                    backend.show()?;
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
                                let listen_ms =
                                    listen_override_ms().unwrap_or(settings.listen_ms as u64);
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
                                if !matches!(gaze_drive, GazeDrive::Mouse | GazeDrive::Chase { .. })
                                {
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
                    let hint = if backend.kind() == BackendKind::Wayland {
                        "move the mouse (Hyprland IPC) or SUPER+B / `buckyboi --wake`"
                    } else {
                        "move the mouse or press a key to return"
                    };
                    eprintln!("buckyboi: hidden ({hint})");
                    backend.hide()?;
                    mapped = false;
                }
            }
        }

        if mapped && ux.phase != Phase::Hidden {
            let listening = ux.phase == Phase::Listening;
            let bounds = overlay_bounds(
                state.cx, state.cy, pad, listening, &menu, now, sw as f32, sh as f32,
            );
            let rect = bounds_rect(bounds, sw, sh);
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
            let pixels = paint_rect(
                &state, listening, pulse, dwell, &menu, &settings, &hud, mx, my, now, sw as f32,
                sh as f32, rect,
            );
            let hits = hit_rects(
                state.cx, state.cy, pad, listening, &menu, now, sw as f32, sh as f32,
            );
            backend.present(&pixels, rect, &hits)?;
        }

        prev_button = button;
        backend.flush()?;

        let spent = tick_start.elapsed();
        let budget = if ux.phase == Phase::Hidden {
            Duration::from_millis(40)
        } else {
            frame
        };
        if let Some(rest) = budget.checked_sub(spent) {
            std::thread::sleep(rest.min(Duration::from_millis(16)));
        }
    }

    backend.shutdown()?;
    eprintln!("buckyboi: quit");
    Ok(())
}
