//! Fullscreen overlay buddy. Wayland (wlr-layer-shell) or X11 Shape/XFixes.
//! See README.md and UX.md.

use buckyboi::display::{
    bounds_rect, open_backend, paint_rect, send_command, BackendKind, FrameInput, WakeBus,
    WakeCommand,
};
use buckyboi::face_to_screen;
use buckyboi::identity::{
    env_flag, env_flag_alias, env_or_alias, hands, latest_face, latest_hand, latest_look,
    latest_voice, reload_rec, set_enrolling, set_gallery_kinds, set_screen, skip_on_enroll_fail,
    start_mic, start_vision_worker, voice, watch_vision_worker, wizard_auto_skip, FirstRunWizard,
    HandStatus, WizardEvent, LARGEST_FACE_CHIP, VOICE_ENROLL_TIMEOUT_MS,
};
use buckyboi::{
    camera, chase_gaze, click_listening_ex, corner_on, gaze_over_hysteresis, hit_rects, hit_test,
    initial_on, load_settings, overlay_bounds, save_settings, spin, step_overlay_avoid,
    AuthSession, BuddyUx, EnrollKind, EnrollPhase, EnrollSession, GateMode, GazeLock, GazeSmoother,
    GestureAction, GestureClass, IdentityHud, Phase, ProfileStore, RadialAction, RadialMenu,
    Settings, SettingsPage, UxEvent, WizardHits, HIT_RADIUS,
};
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SNAP_FRESH_MS: u64 = 400;
const GESTURE_ACTION_MS: u64 = 480;

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
    wizard: &FirstRunWizard,
    now: u64,
) -> IdentityHud {
    let face_chip = if wizard.using_largest_face() {
        LARGEST_FACE_CHIP.into()
    } else if latest_look(now, 400).map(|l| l.faces).unwrap_or(0) > 1 {
        LARGEST_FACE_CHIP.into()
    } else {
        String::new()
    };
    IdentityHud {
        auth_label: auth.state.label(now),
        people: profiles.people().iter().map(|p| p.name.clone()).collect(),
        enroll_hint: enroll.hint(),
        enroll_progress: enroll.progress(),
        enroll_active: !matches!(enroll.phase, EnrollPhase::Idle) && !wizard.is_open(),
        wizard_open: wizard.is_open(),
        wizard_title: wizard.title().into(),
        wizard_body: wizard.body(enroll),
        wizard_progress: wizard.progress(enroll),
        wizard_can_skip: wizard.can_skip(),
        wizard_can_next: wizard.can_next(),
        wizard_can_add: wizard.can_add_another(),
        face_chip,
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
            set_gallery_kinds(&profiles.face_kinds());
            reload_rec();
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

fn enroll_capturing(enroll: &EnrollSession, kind: EnrollKind) -> bool {
    enroll.kind == kind
        && !matches!(
            enroll.phase,
            EnrollPhase::Idle | EnrollPhase::Done { .. } | EnrollPhase::Failed { .. }
        )
}

fn identity_tick(
    now: u64,
    profiles: &mut ProfileStore,
    auth: &mut AuthSession,
    enroll: &mut EnrollSession,
    wizard: &mut FirstRunWizard,
    settings: &Settings,
    last_face_ms: &mut u64,
    last_voice_ms: &mut u64,
    last_hand_ms: &mut u64,
) {
    auth.expire(now);
    set_enrolling(EnrollKind::Face, enroll_capturing(enroll, EnrollKind::Face));
    set_enrolling(
        EnrollKind::Voice,
        enroll_capturing(enroll, EnrollKind::Voice),
    );

    if env_flag("BUCKYBOI_FACE_SIM") {
        let (rgb, w, h) = synthetic_skin();
        camera::store_frame(camera::CamFrame {
            rgb: Arc::from(rgb),
            w,
            h,
            t_ms: now,
        });
    }

    if let Some(snap) = latest_face() {
        if now.saturating_sub(snap.t_ms) <= SNAP_FRESH_MS && snap.t_ms != *last_face_ms {
            *last_face_ms = snap.t_ms;
            wizard.note_faces(snap.quality.faces);
            if enroll_capturing(enroll, EnrollKind::Face) {
                let _ = enroll.push_face(snap.quality, snap.embedding);
            } else if let Some(emb) = snap.embedding {
                if let Some(hit) = profiles.match_face(&emb) {
                    auth.note_face(&hit.person_id, &hit.name, now, hit.score);
                }
            }
        }
    }

    if let Some(snap) = latest_voice() {
        if now.saturating_sub(snap.t_ms) <= SNAP_FRESH_MS && snap.t_ms != *last_voice_ms {
            *last_voice_ms = snap.t_ms;
            if enroll_capturing(enroll, EnrollKind::Voice) {
                let _ = enroll.push_voice(snap.quality, snap.embedding);
            } else if snap.quality.ok {
                if let Some(emb) = snap.embedding {
                    if let Some(hit) = profiles.match_voice(&emb) {
                        auth.note_voice(&hit.person_id, &hit.name, now, hit.score);
                    }
                }
            }
        }
    }

    if enroll_capturing(enroll, EnrollKind::Voice) && enroll.timed_out(now, VOICE_ENROLL_TIMEOUT_MS)
    {
        enroll.fail_now("TIMEOUT");
    }

    if enroll_capturing(enroll, EnrollKind::Gesture) {
        match latest_hand() {
            Some(snap)
                if now.saturating_sub(snap.t_ms) <= SNAP_FRESH_MS && snap.t_ms != *last_hand_ms =>
            {
                *last_hand_ms = snap.t_ms;
                let _ = enroll.apply_hand_status(&snap.status);
            }
            _ => {}
        }
    }

    let _ = settings;
}

fn apply_identity_action(
    act: RadialAction,
    settings: &mut Settings,
    profiles: &mut ProfileStore,
    enroll: &mut EnrollSession,
    auth: &mut AuthSession,
    now: u64,
) {
    match act {
        RadialAction::CycleGate
        | RadialAction::ToggleNeedFace
        | RadialAction::TabLook
        | RadialAction::TabPeople => {
            save_settings(settings);
        }
        RadialAction::NewPerson => {
            // Legacy: empty row. ADD now opens the guided wizard.
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
            enroll.begin_capture_at(now);
            eprintln!("buckyboi: face enroll — look at the camera (or BUCKYBOI_FACE_SIM=1)");
        }
        RadialAction::StartEnrollVoice => {
            let (name, id) = person_target(profiles, settings);
            *enroll = EnrollSession::start_voice(name, id);
            if !voice::input_available() {
                enroll.fail_now("NO MIC");
                eprintln!("buckyboi: voice enroll failed (NO MIC)");
            } else {
                enroll.begin_capture_at(now);
                eprintln!("buckyboi: voice enroll — speak for ~1.5s × 3 (or BUCKYBOI_VOICE_SIM=1)");
            }
        }
        RadialAction::StartEnrollHands => {
            let (name, id) = person_target(profiles, settings);
            *enroll = EnrollSession::start_gesture(
                name,
                id,
                GestureClass::Fist,
                hands::GESTURE_ENROLL_NEED,
            );
            enroll.begin_capture_at(now);
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

fn wizard_hits(wizard: &FirstRunWizard) -> Option<WizardHits> {
    wizard.is_open().then_some(WizardHits {
        can_skip: wizard.can_skip(),
        can_next: wizard.can_next(),
        can_add: wizard.can_add_another(),
    })
}

fn show_wizard(
    wizard: &mut FirstRunWizard,
    menu: &mut RadialMenu,
    ux: &mut BuddyUx,
    profiles: &ProfileStore,
    settings: &Settings,
    now: u64,
    first_run: bool,
) {
    let name = profiles.next_default_name();
    *wizard = FirstRunWizard::open(name, first_run);
    menu.wizard_open = true;
    menu.settings_open = true;
    menu.freeze_ms = Some(now);
    if ux.phase != Phase::Listening {
        let listen_ms = listen_override_ms().unwrap_or(settings.listen_ms as u64);
        ux.enter_listen_for(now, listen_ms);
    }
    eprintln!(
        "buckyboi: {} wizard for {} — NEXT to enroll",
        wizard.title(),
        wizard.person_name
    );
}

fn close_wizard_chrome(menu: &mut RadialMenu) {
    menu.wizard_open = false;
    menu.settings_open = false;
    menu.freeze_ms = None;
}

fn ensure_wizard_person(
    wizard: &mut FirstRunWizard,
    profiles: &mut ProfileStore,
    settings: &mut Settings,
) {
    if wizard.person_id.is_some() {
        return;
    }
    let id = profiles.upsert_named(&wizard.person_name);
    profiles.save();
    wizard.person_id = Some(id.clone());
    if let Some(i) = profiles.people().iter().position(|p| p.id == id) {
        settings.selected_person = i;
    }
}

fn stamp_wizard_auth(wizard: &FirstRunWizard, auth: &mut AuthSession, now: u64) {
    let (Some(id), name) = (wizard.person_id.as_deref(), wizard.person_name.as_str()) else {
        return;
    };
    if wizard.face_ok {
        auth.note_face(id, name, now, 1.0);
    }
    if wizard.voice_ok {
        auth.note_voice(id, name, now, 1.0);
    }
}

fn apply_wizard_event(
    ev: WizardEvent,
    wizard: &mut FirstRunWizard,
    enroll: &mut EnrollSession,
    profiles: &mut ProfileStore,
    settings: &mut Settings,
    menu: &mut RadialMenu,
    now: u64,
) {
    match ev {
        WizardEvent::None | WizardEvent::Opened => {}
        WizardEvent::BeginFace => {
            ensure_wizard_person(wizard, profiles, settings);
            *enroll =
                EnrollSession::start_face(wizard.person_name.clone(), wizard.person_id.clone());
            enroll.begin_capture_at(now);
            eprintln!("buckyboi: wizard FACE — look at the camera (or BUCKYBOI_FACE_SIM=1)");
        }
        WizardEvent::BeginVoice => {
            ensure_wizard_person(wizard, profiles, settings);
            *enroll =
                EnrollSession::start_voice(wizard.person_name.clone(), wizard.person_id.clone());
            if !voice::input_available() {
                enroll.fail_now("NO MIC");
                eprintln!("buckyboi: wizard VOICE failed (NO MIC)");
            } else {
                enroll.begin_capture_at(now);
                eprintln!(
                    "buckyboi: wizard VOICE — speak ~1.5s × 3 or SKIP (BUCKYBOI_VOICE_SIM=1)"
                );
            }
        }
        WizardEvent::BeginHands(class) => {
            ensure_wizard_person(wizard, profiles, settings);
            *enroll = EnrollSession::start_gesture(
                wizard.person_name.clone(),
                wizard.person_id.clone(),
                class,
                hands::GESTURE_ENROLL_NEED,
            );
            enroll.begin_capture_at(now);
            eprintln!(
                "buckyboi: wizard HAND {} — hold or SKIP (BUCKYBOI_HAND_SIM)",
                class.label()
            );
        }
        WizardEvent::SuggestGate(g) => {
            if settings.gate == GateMode::Off {
                settings.gate = g;
                save_settings(settings);
                eprintln!("buckyboi: gate → {} (wizard suggestion)", g.label());
            }
            if !wizard.is_open() {
                close_wizard_chrome(menu);
                eprintln!("buckyboi: wizard done");
            }
        }
        WizardEvent::Completed => {
            close_wizard_chrome(menu);
            eprintln!("buckyboi: wizard done");
        }
        WizardEvent::Cancelled => {
            enroll.cancel();
            close_wizard_chrome(menu);
            eprintln!("buckyboi: wizard cancelled");
        }
    }
}

fn settle_enroll(
    now: u64,
    enroll: &mut EnrollSession,
    wizard: &mut FirstRunWizard,
    profiles: &mut ProfileStore,
    settings: &mut Settings,
    menu: &mut RadialMenu,
) {
    for _ in 0..8 {
        if matches!(enroll.phase, EnrollPhase::Done { .. }) {
            let kind = enroll.kind;
            finish_enroll(enroll, profiles, settings);
            if wizard.is_open() {
                apply_wizard_event(
                    wizard.on_enroll_finished(kind),
                    wizard,
                    enroll,
                    profiles,
                    settings,
                    menu,
                    now,
                );
            }
            continue;
        }
        if let EnrollPhase::Failed { reason } = &enroll.phase {
            let reason = reason.clone();
            let kind = enroll.kind;
            if wizard.is_open()
                && (wizard_auto_skip(kind, &reason) || skip_on_enroll_fail(kind, &reason))
            {
                *enroll = EnrollSession::idle();
                apply_wizard_event(wizard.skip(), wizard, enroll, profiles, settings, menu, now);
                continue;
            }
            // Leave Failed so the HUD can show NO MIC / TIMEOUT / NO FACE.
        }
        break;
    }
}

fn maybe_offer_wizard(
    wizard: &mut FirstRunWizard,
    menu: &mut RadialMenu,
    ux: &mut BuddyUx,
    profiles: &ProfileStore,
    settings: &Settings,
    now: u64,
) -> bool {
    if wizard.is_open() {
        return false;
    }
    if FirstRunWizard::should_offer(settings.gate, profiles.enrolled_count()) {
        show_wizard(wizard, menu, ux, profiles, settings, now, true);
        return true;
    }
    false
}

fn maybe_gesture_action(
    now: u64,
    profiles: &ProfileStore,
    auth: &AuthSession,
    settings: &Settings,
    last_gesture_action_ms: &mut u64,
) -> Option<GestureAction> {
    if now.saturating_sub(*last_gesture_action_ms) < GESTURE_ACTION_MS {
        return None;
    }
    let snap = latest_hand()?;
    if now.saturating_sub(snap.t_ms) > SNAP_FRESH_MS {
        return None;
    }
    let HandStatus::Ok(hand) = snap.status else {
        return None;
    };
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
    *last_gesture_action_ms = now;
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
         listen if the gate allows. Empty gallery + gate on opens a first-run\n\
         enroll wizard. Settings → ID still enrolls one modality at a time.\n\
         Profiles stay in ~/.config/buckyboi/ (offline).\n\n\
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
    start_mic();
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
    buckyboi::identity::face::set_gallery_kinds(&profiles.face_kinds());
    profiles.file.face_threshold = profiles.file.face_threshold.max(0.15);
    start_vision_worker(&profiles.face_kinds());
    set_screen(sw as u32, sh as u32);
    let mut auth = AuthSession::new(8_000);
    let mut enroll = EnrollSession::idle();
    let mut wizard = FirstRunWizard::closed();
    let mut wizard_declined = false;
    let mut last_hand_ms = 0u64;
    let mut last_voice_ms = 0u64;
    let mut last_face_ms = 0u64;
    let mut last_gesture_action_ms = 0u64;

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
    let mut prev_escape = false;
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
    if FirstRunWizard::should_offer(settings.gate, profiles.enrolled_count()) {
        show_wizard(
            &mut wizard,
            &mut menu,
            &mut ux,
            &profiles,
            &settings,
            now_ms(),
            true,
        );
    }
    loop {
        let tick_start = Instant::now();
        let (nw, nh) = backend.size();
        if nw != sw || nh != sh {
            sw = nw;
            sh = nh;
            state.sw = sw as f32;
            state.sh = sh as f32;
            set_screen(sw as u32, sh as u32);
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

        let now = now_ms();
        if input.escape && !prev_escape {
            if wizard.is_open() {
                wizard_declined = true;
                let ev = wizard.cancel();
                apply_wizard_event(
                    ev,
                    &mut wizard,
                    &mut enroll,
                    &mut profiles,
                    &mut settings,
                    &mut menu,
                    now,
                );
            } else {
                break;
            }
        }
        prev_escape = input.escape;

        watch_vision_worker(now);
        identity_tick(
            now,
            &mut profiles,
            &mut auth,
            &mut enroll,
            &mut wizard,
            &settings,
            &mut last_face_ms,
            &mut last_voice_ms,
            &mut last_hand_ms,
        );
        settle_enroll(
            now,
            &mut enroll,
            &mut wizard,
            &mut profiles,
            &mut settings,
            &mut menu,
        );
        if matches!(enroll.phase, EnrollPhase::Idle) && !wizard.is_open() {
            if let Some(gact) = maybe_gesture_action(
                now,
                &profiles,
                &auth,
                &settings,
                &mut last_gesture_action_ms,
            ) {
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
                        look.cx, look.cy, look.fw, look.fh, sw as f32, sh as f32, true,
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
                                if !wizard_declined
                                    && maybe_offer_wizard(
                                        &mut wizard,
                                        &mut menu,
                                        &mut ux,
                                        &profiles,
                                        &settings,
                                        now,
                                    )
                                {
                                    eprintln!("buckyboi: listen needs enroll — opening wizard");
                                } else {
                                    eprintln!(
                                        "buckyboi: listen blocked (gate {} / {})",
                                        settings.gate.label(),
                                        auth.state.label(now)
                                    );
                                }
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
                                if !wizard_declined
                                    && maybe_offer_wizard(
                                        &mut wizard,
                                        &mut menu,
                                        &mut ux,
                                        &profiles,
                                        &settings,
                                        now,
                                    )
                                {
                                    eprintln!("buckyboi: gaze-lock needs enroll — opening wizard");
                                } else {
                                    eprintln!(
                                        "buckyboi: gaze-lock ignored (gate {} / {})",
                                        settings.gate.label(),
                                        auth.state.label(now)
                                    );
                                }
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
                        !matches!(enroll.phase, EnrollPhase::Idle) && !wizard.is_open(),
                        wizard_hits(&wizard),
                    );
                    apply_identity_action(
                        act,
                        &mut settings,
                        &mut profiles,
                        &mut enroll,
                        &mut auth,
                        now,
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
                            } else {
                                camera::request_stop();
                                if matches!(gaze_drive, GazeDrive::Camera(_)) {
                                    gaze_drive = GazeDrive::Off;
                                }
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
                        RadialAction::CycleGate => {
                            if maybe_offer_wizard(
                                &mut wizard,
                                &mut menu,
                                &mut ux,
                                &profiles,
                                &settings,
                                now,
                            ) {
                                wizard_declined = false;
                                eprintln!("buckyboi: gate on + empty gallery — enroll wizard");
                            }
                        }
                        RadialAction::OpenWizard => {
                            wizard_declined = false;
                            show_wizard(
                                &mut wizard,
                                &mut menu,
                                &mut ux,
                                &profiles,
                                &settings,
                                now,
                                false,
                            );
                        }
                        RadialAction::WizardAdvance => {
                            stamp_wizard_auth(&wizard, &mut auth, now);
                            let ev = wizard.advance();
                            apply_wizard_event(
                                ev,
                                &mut wizard,
                                &mut enroll,
                                &mut profiles,
                                &mut settings,
                                &mut menu,
                                now,
                            );
                        }
                        RadialAction::WizardSkip => {
                            enroll.cancel();
                            let ev = wizard.skip();
                            apply_wizard_event(
                                ev,
                                &mut wizard,
                                &mut enroll,
                                &mut profiles,
                                &mut settings,
                                &mut menu,
                                now,
                            );
                        }
                        RadialAction::WizardCancel => {
                            wizard_declined = true;
                            let ev = wizard.cancel();
                            apply_wizard_event(
                                ev,
                                &mut wizard,
                                &mut enroll,
                                &mut profiles,
                                &mut settings,
                                &mut menu,
                                now,
                            );
                        }
                        RadialAction::WizardAddAnother => {
                            let name = profiles.next_default_name();
                            enroll.cancel();
                            let ev = wizard.add_another(name);
                            apply_wizard_event(
                                ev,
                                &mut wizard,
                                &mut enroll,
                                &mut profiles,
                                &mut settings,
                                &mut menu,
                                now,
                            );
                        }
                        RadialAction::None
                        | RadialAction::ToggleNeedFace
                        | RadialAction::TabLook
                        | RadialAction::TabPeople
                        | RadialAction::NewPerson
                        | RadialAction::SelectPerson(_)
                        | RadialAction::DeleteSelected
                        | RadialAction::CancelEnroll => {}
                    }
                }
                if !menu.panel_open() && ux.tick(now) == UxEvent::Hide {
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

        settle_enroll(
            now,
            &mut enroll,
            &mut wizard,
            &mut profiles,
            &mut settings,
            &mut menu,
        );

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
            let hud = hud_from(&auth, &profiles, &settings, &enroll, &wizard, now);
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
