//! Listening-mode radial icons + attached settings panel (pure logic).

use crate::identity::{identity_ini_lines, parse_identity_ini, GateMode, IdentitySettings};
use std::fs;
use std::path::PathBuf;

pub const ICON_COUNT: usize = 4;
pub const ICON_RADIUS: f32 = 17.0;
pub const ORBIT_RADIUS: f32 = 122.0;
pub const PANEL_W: f32 = 268.0;
pub const PANEL_H: f32 = 300.0;
pub const CLOSE_R: f32 = 9.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SettingsPage {
    #[default]
    Look,
    People,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IconId {
    Settings,
    Dismiss,
    Mute,
    Info,
}

pub const ICONS: [IconId; ICON_COUNT] = [
    IconId::Settings,
    IconId::Dismiss,
    IconId::Mute,
    IconId::Info,
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RadialAction {
    None,
    ToggleSettings,
    CloseSettings,
    Dismiss,
    Mute,
    Info,
    SetListenMs(u32),
    SetGazeMs(u32),
    SetStroke(f32),
    ToggleCamera,
    CycleGate,
    ToggleNeedFace,
    TabLook,
    TabPeople,
    NewPerson,
    SelectPerson(usize),
    DeleteSelected,
    StartEnrollFace,
    StartEnrollVoice,
    StartEnrollHands,
    CancelEnroll,
    OpenWizard,
    WizardAdvance,
    WizardSkip,
    WizardCancel,
    WizardAddAnother,
    StartGazeCalib,
    CancelGazeCalib,
    SkipGazeCalib,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub listen_ms: u32,
    pub gaze_lock_ms: u32,
    pub camera: bool,
    pub stroke: f32,
    pub gate: GateMode,
    pub gestures_need_face: bool,
    pub page: SettingsPage,
    pub selected_person: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            listen_ms: 8_000,
            gaze_lock_ms: 8_000,
            camera: true,
            stroke: 1.45,
            gate: GateMode::Off,
            gestures_need_face: true,
            page: SettingsPage::Look,
            selected_person: 0,
        }
    }
}

impl Settings {
    pub fn clamp(&mut self) {
        self.listen_ms = self.listen_ms.clamp(2_000, 15_000);
        self.gaze_lock_ms = self.gaze_lock_ms.clamp(2_000, 15_000);
        self.stroke = self.stroke.clamp(0.8, 3.0);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WizardHits {
    pub can_skip: bool,
    pub can_next: bool,
    pub can_add: bool,
}

#[derive(Clone, Debug, Default)]
pub struct IdentityHud {
    pub auth_label: String,
    pub people: Vec<String>,
    pub enroll_hint: String,
    pub enroll_progress: f32,
    pub enroll_active: bool,
    pub wizard_open: bool,
    pub wizard_title: String,
    pub wizard_body: String,
    pub wizard_progress: f32,
    pub wizard_can_skip: bool,
    pub wizard_can_next: bool,
    pub wizard_can_add: bool,
    pub face_chip: String,
    pub gaze_calib_open: bool,
    pub gaze_calib_hint: String,
    pub gaze_calib_progress: f32,
}

#[derive(Clone, Debug)]
pub struct RadialMenu {
    pub settings_open: bool,
    pub wizard_open: bool,
    pub muted: bool,
    pub flash_id: Option<IconId>,
    pub flash_until_ms: u64,
    pub dragging_slider: Option<Slider>,
    pub freeze_ms: Option<u64>,
    pub calib_open: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slider {
    Listen,
    Gaze,
    Stroke,
}

impl RadialMenu {
    pub fn new() -> Self {
        Self {
            settings_open: false,
            wizard_open: false,
            muted: false,
            flash_id: None,
            flash_until_ms: 0,
            dragging_slider: None,
            freeze_ms: None,
            calib_open: false,
        }
    }

    pub fn reset(&mut self) {
        self.settings_open = false;
        self.wizard_open = false;
        self.flash_id = None;
        self.dragging_slider = None;
        self.freeze_ms = None;
        self.calib_open = false;
    }

    /// Settings, first-run wizard, or gaze calib — hide-timer pauses.
    pub fn panel_open(&self) -> bool {
        self.settings_open || self.wizard_open || self.calib_open
    }

    pub fn layout_ms(&self, now_ms: u64) -> u64 {
        self.freeze_ms.unwrap_or(now_ms)
    }

    pub fn flash(&mut self, id: IconId, now_ms: u64) {
        self.flash_id = Some(id);
        self.flash_until_ms = now_ms + 220;
    }

    pub fn flashing(&self, id: IconId, now_ms: u64) -> bool {
        self.flash_id == Some(id) && now_ms < self.flash_until_ms
    }
}

impl Default for RadialMenu {
    fn default() -> Self {
        Self::new()
    }
}

/// Icon centers. Settings sits upper-right of the buddy; others follow clockwise.
/// Slow orbit unless the settings panel is open (stable click targets).
pub fn icon_centers(
    cx: f32,
    cy: f32,
    now_ms: u64,
    orbit: bool,
) -> [(IconId, f32, f32); ICON_COUNT] {
    let spin = if orbit {
        (now_ms as f32 / 2800.0) % std::f32::consts::TAU
    } else {
        0.0
    };
    let mut out = [(IconId::Settings, 0.0, 0.0); ICON_COUNT];
    // Nudge the orbit toward the screen interior when the buddy is in a corner.
    let ox = if cx < 180.0 {
        38.0
    } else if cx > 1600.0 {
        -38.0
    } else {
        0.0
    };
    let oy = if cy < 180.0 {
        38.0
    } else if cy > 900.0 {
        -38.0
    } else {
        0.0
    };
    for (i, id) in ICONS.iter().enumerate() {
        // Start at +X so the home (top-left) lock keeps every disc on-screen.
        let a = spin + i as f32 * (std::f32::consts::TAU / ICON_COUNT as f32);
        out[i] = (
            *id,
            cx + ox + ORBIT_RADIUS * a.cos(),
            cy + oy + ORBIT_RADIUS * a.sin(),
        );
    }
    out
}

pub fn hit_icon(mx: f32, my: f32, centers: &[(IconId, f32, f32)]) -> Option<IconId> {
    let mut best = None;
    let mut best_d = ICON_RADIUS;
    for &(id, x, y) in centers {
        let d = (mx - x).hypot(my - y);
        if d <= best_d {
            best_d = d;
            best = Some(id);
        }
    }
    best
}

/// Panel anchored to the Settings icon, pushed onto the screen.
pub fn panel_rect(
    settings_x: f32,
    settings_y: f32,
    screen_w: f32,
    screen_h: f32,
) -> (f32, f32, f32, f32) {
    let mut x = settings_x + ICON_RADIUS + 10.0;
    let mut y = settings_y - 18.0;
    if x + PANEL_W > screen_w - 8.0 {
        x = settings_x - ICON_RADIUS - 10.0 - PANEL_W;
    }
    if y + PANEL_H > screen_h - 8.0 {
        y = screen_h - 8.0 - PANEL_H;
    }
    if x < 8.0 {
        x = 8.0;
    }
    if y < 8.0 {
        y = 8.0;
    }
    (x, y, PANEL_W, PANEL_H)
}

pub fn close_button(px: f32, py: f32) -> (f32, f32) {
    (px + PANEL_W - 16.0, py + 16.0)
}

pub(crate) fn slider_track(px: f32, py: f32, row: usize) -> (f32, f32, f32, f32) {
    let y = py + 92.0 + row as f32 * 28.0;
    (px + 88.0, y, PANEL_W - 104.0, 10.0)
}

pub(crate) fn camera_hit(px: f32, py: f32) -> (f32, f32, f32, f32) {
    (px + 88.0, py + 64.0, 72.0, 16.0)
}

pub(crate) fn tab_look_hit(px: f32, py: f32) -> (f32, f32, f32, f32) {
    (px + 10.0, py + 32.0, 52.0, 16.0)
}

pub(crate) fn tab_people_hit(px: f32, py: f32) -> (f32, f32, f32, f32) {
    (px + 68.0, py + 32.0, 36.0, 16.0)
}

pub(crate) fn gate_hit(px: f32, py: f32) -> (f32, f32, f32, f32) {
    (px + 168.0, py + 64.0, 86.0, 16.0)
}

pub(crate) fn person_row_hit(px: f32, py: f32, i: usize) -> (f32, f32, f32, f32) {
    (px + 10.0, py + 88.0 + i as f32 * 20.0, PANEL_W - 20.0, 18.0)
}

pub(crate) fn id_btn(px: f32, py: f32, col: usize) -> (f32, f32, f32, f32) {
    let x = px + 10.0 + col as f32 * 50.0;
    (x, py + 196.0, 46.0, 16.0)
}

pub(crate) fn new_btn(px: f32, py: f32) -> (f32, f32, f32, f32) {
    (px + 10.0, py + 220.0, 52.0, 16.0)
}

pub(crate) fn del_btn(px: f32, py: f32) -> (f32, f32, f32, f32) {
    (px + 68.0, py + 220.0, 46.0, 16.0)
}

pub(crate) fn need_face_hit(px: f32, py: f32) -> (f32, f32, f32, f32) {
    (px + 10.0, py + 248.0, 150.0, 16.0)
}

pub(crate) fn cancel_enroll_hit(px: f32, py: f32) -> (f32, f32, f32, f32) {
    (px + 10.0, py + 268.0, 70.0, 16.0)
}

pub(crate) fn wizard_next_hit(px: f32, py: f32) -> (f32, f32, f32, f32) {
    id_btn(px, py, 0)
}

pub(crate) fn wizard_skip_hit(px: f32, py: f32) -> (f32, f32, f32, f32) {
    id_btn(px, py, 1)
}

pub(crate) fn wizard_add_hit(px: f32, py: f32) -> (f32, f32, f32, f32) {
    new_btn(px, py)
}

/// CALIB row under the three Look-tab sliders (do not squeeze next to CAM/GATE).
pub(crate) fn calib_hit(px: f32, py: f32) -> (f32, f32, f32, f32) {
    (px + 10.0, py + 92.0 + 3.0 * 28.0 + 8.0, 70.0, 16.0)
}

/// SKIP / CANCEL discs sit just under the buddy during calib (17 px).
pub fn calib_chip_centers(cx: f32, cy: f32) -> ((f32, f32), (f32, f32)) {
    ((cx - 48.0, cy + 118.0), (cx + 48.0, cy + 118.0))
}

fn in_rect(mx: f32, my: f32, x: f32, y: f32, w: f32, h: f32) -> bool {
    mx >= x && my >= y && mx <= x + w && my <= y + h
}

fn slider_value(mx: f32, track: (f32, f32, f32, f32), min: f32, max: f32) -> f32 {
    let t = ((mx - track.0) / track.2).clamp(0.0, 1.0);
    min + t * (max - min)
}

pub fn hit_panel(mx: f32, my: f32, px: f32, py: f32, settings: &Settings) -> RadialAction {
    hit_panel_ex(mx, my, px, py, settings, 0, false, None)
}

pub fn hit_panel_ex(
    mx: f32,
    my: f32,
    px: f32,
    py: f32,
    settings: &Settings,
    people_n: usize,
    enroll_active: bool,
    wizard: Option<WizardHits>,
) -> RadialAction {
    let (cx, cy) = close_button(px, py);
    if (mx - cx).hypot(my - cy) <= CLOSE_R {
        if wizard.is_some() {
            return RadialAction::WizardCancel;
        }
        return RadialAction::CloseSettings;
    }
    if let Some(wiz) = wizard {
        if wiz.can_next {
            let n = wizard_next_hit(px, py);
            if in_rect(mx, my, n.0, n.1, n.2, n.3) {
                return RadialAction::WizardAdvance;
            }
        }
        if wiz.can_skip {
            let s = wizard_skip_hit(px, py);
            if in_rect(mx, my, s.0, s.1, s.2, s.3) {
                return RadialAction::WizardSkip;
            }
        }
        if wiz.can_add {
            let a = wizard_add_hit(px, py);
            if in_rect(mx, my, a.0, a.1, a.2, a.3) {
                return RadialAction::WizardAddAnother;
            }
        }
        if in_rect(mx, my, px, py, PANEL_W, PANEL_H) {
            return RadialAction::None;
        }
        return RadialAction::None;
    }
    let look = tab_look_hit(px, py);
    if in_rect(mx, my, look.0, look.1, look.2, look.3) {
        return RadialAction::TabLook;
    }
    let people = tab_people_hit(px, py);
    if in_rect(mx, my, people.0, people.1, people.2, people.3) {
        return RadialAction::TabPeople;
    }
    if enroll_active {
        let c = cancel_enroll_hit(px, py);
        if in_rect(mx, my, c.0, c.1, c.2, c.3) {
            return RadialAction::CancelEnroll;
        }
        if in_rect(mx, my, px, py, PANEL_W, PANEL_H) {
            return RadialAction::None;
        }
        return RadialAction::None;
    }
    if settings.page == SettingsPage::People {
        let n = people_n.min(4);
        for i in 0..n {
            let r = person_row_hit(px, py, i);
            if in_rect(mx, my, r.0, r.1, r.2, r.3) {
                return RadialAction::SelectPerson(i);
            }
        }
        for (col, act) in [
            (0, RadialAction::StartEnrollFace),
            (1, RadialAction::StartEnrollVoice),
            (2, RadialAction::StartEnrollHands),
        ] {
            let b = id_btn(px, py, col);
            if in_rect(mx, my, b.0, b.1, b.2, b.3) {
                return act;
            }
        }
        let nb = new_btn(px, py);
        if in_rect(mx, my, nb.0, nb.1, nb.2, nb.3) {
            return RadialAction::OpenWizard;
        }
        let db = del_btn(px, py);
        if in_rect(mx, my, db.0, db.1, db.2, db.3) {
            return RadialAction::DeleteSelected;
        }
        let nf = need_face_hit(px, py);
        if in_rect(mx, my, nf.0, nf.1, nf.2, nf.3) {
            return RadialAction::ToggleNeedFace;
        }
        if in_rect(mx, my, px, py, PANEL_W, PANEL_H) {
            return RadialAction::None;
        }
        return RadialAction::None;
    }
    let cam = camera_hit(px, py);
    if in_rect(mx, my, cam.0, cam.1, cam.2, cam.3) {
        return RadialAction::ToggleCamera;
    }
    let gate = gate_hit(px, py);
    if in_rect(mx, my, gate.0, gate.1, gate.2, gate.3) {
        return RadialAction::CycleGate;
    }
    let listen = slider_track(px, py, 0);
    if in_rect(
        mx,
        my,
        listen.0 - 4.0,
        listen.1 - 6.0,
        listen.2 + 8.0,
        listen.3 + 12.0,
    ) {
        let v = slider_value(mx, listen, 2_000.0, 15_000.0) as u32;
        return RadialAction::SetListenMs(v);
    }
    let gaze = slider_track(px, py, 1);
    if in_rect(
        mx,
        my,
        gaze.0 - 4.0,
        gaze.1 - 6.0,
        gaze.2 + 8.0,
        gaze.3 + 12.0,
    ) {
        return RadialAction::SetGazeMs(slider_value(mx, gaze, 2_000.0, 15_000.0) as u32);
    }
    let stroke = slider_track(px, py, 2);
    if in_rect(
        mx,
        my,
        stroke.0 - 4.0,
        stroke.1 - 6.0,
        stroke.2 + 8.0,
        stroke.3 + 12.0,
    ) {
        return RadialAction::SetStroke(slider_value(mx, stroke, 0.8, 3.0));
    }
    let calib = calib_hit(px, py);
    if in_rect(mx, my, calib.0, calib.1, calib.2, calib.3) {
        return RadialAction::StartGazeCalib;
    }
    if in_rect(mx, my, px, py, PANEL_W, PANEL_H) {
        return RadialAction::None;
    }
    RadialAction::None
}

pub fn apply_action(
    menu: &mut RadialMenu,
    settings: &mut Settings,
    action: RadialAction,
    now_ms: u64,
) {
    match action {
        RadialAction::ToggleSettings => {
            menu.settings_open = !menu.settings_open;
            menu.freeze_ms = if menu.settings_open {
                Some(now_ms)
            } else {
                None
            };
            menu.flash(IconId::Settings, now_ms);
        }
        RadialAction::CloseSettings => {
            menu.settings_open = false;
            menu.freeze_ms = None;
        }
        RadialAction::Dismiss => {
            menu.flash(IconId::Dismiss, now_ms);
            menu.reset();
        }
        RadialAction::Mute => {
            menu.muted = !menu.muted;
            menu.flash(IconId::Mute, now_ms);
        }
        RadialAction::Info => {
            menu.flash(IconId::Info, now_ms);
        }
        RadialAction::SetListenMs(v) => {
            settings.listen_ms = v;
            settings.clamp();
        }
        RadialAction::SetGazeMs(v) => {
            settings.gaze_lock_ms = v;
            settings.clamp();
        }
        RadialAction::SetStroke(v) => {
            settings.stroke = v;
            settings.clamp();
        }
        RadialAction::ToggleCamera => {
            settings.camera = !settings.camera;
        }
        RadialAction::CycleGate => {
            settings.gate = settings.gate.next();
        }
        RadialAction::ToggleNeedFace => {
            settings.gestures_need_face = !settings.gestures_need_face;
        }
        RadialAction::TabLook => {
            settings.page = SettingsPage::Look;
        }
        RadialAction::TabPeople => {
            settings.page = SettingsPage::People;
        }
        RadialAction::SelectPerson(i) => {
            settings.selected_person = i;
        }
        RadialAction::NewPerson
        | RadialAction::OpenWizard
        | RadialAction::DeleteSelected
        | RadialAction::StartEnrollFace
        | RadialAction::StartEnrollVoice
        | RadialAction::StartEnrollHands
        | RadialAction::CancelEnroll
        | RadialAction::WizardAdvance
        | RadialAction::WizardSkip
        | RadialAction::WizardCancel
        | RadialAction::WizardAddAnother
        | RadialAction::StartGazeCalib
        | RadialAction::CancelGazeCalib
        | RadialAction::SkipGazeCalib
        | RadialAction::None => {}
    }
}

pub fn click_listening(
    menu: &mut RadialMenu,
    settings: &mut Settings,
    mx: f32,
    my: f32,
    cx: f32,
    cy: f32,
    now_ms: u64,
    screen_w: f32,
    screen_h: f32,
) -> RadialAction {
    let centers = icon_centers(cx, cy, menu.layout_ms(now_ms), true);
    if menu.settings_open {
        if let Some((_, sx, sy)) = centers.iter().find(|(id, _, _)| *id == IconId::Settings) {
            let (px, py, _, _) = panel_rect(*sx, *sy, screen_w, screen_h);
            let act = hit_panel(mx, my, px, py, settings);
            if act != RadialAction::None {
                apply_action(menu, settings, act, now_ms);
                return act;
            }
        }
    }
    if let Some(id) = hit_icon(mx, my, &centers) {
        let act = match id {
            IconId::Settings => RadialAction::ToggleSettings,
            IconId::Dismiss => RadialAction::Dismiss,
            IconId::Mute => RadialAction::Mute,
            IconId::Info => RadialAction::Info,
        };
        apply_action(menu, settings, act, now_ms);
        return act;
    }
    RadialAction::None
}

pub fn click_listening_ex(
    menu: &mut RadialMenu,
    settings: &mut Settings,
    mx: f32,
    my: f32,
    cx: f32,
    cy: f32,
    now_ms: u64,
    screen_w: f32,
    screen_h: f32,
    people_n: usize,
    enroll_active: bool,
    wizard: Option<WizardHits>,
) -> RadialAction {
    let centers = icon_centers(cx, cy, menu.layout_ms(now_ms), true);
    if menu.calib_open {
        let ((sx, sy), (cx_, cy_)) = calib_chip_centers(cx, cy);
        if (mx - sx).hypot(my - sy) <= ICON_RADIUS {
            return RadialAction::SkipGazeCalib;
        }
        if (mx - cx_).hypot(my - cy_) <= ICON_RADIUS {
            return RadialAction::CancelGazeCalib;
        }
        return RadialAction::None;
    }
    if menu.settings_open || menu.wizard_open {
        if let Some((_, sx, sy)) = centers.iter().find(|(id, _, _)| *id == IconId::Settings) {
            let (px, py, _, _) = panel_rect(*sx, *sy, screen_w, screen_h);
            let act = hit_panel_ex(mx, my, px, py, settings, people_n, enroll_active, wizard);
            if act != RadialAction::None {
                apply_action(menu, settings, act, now_ms);
                return act;
            }
        }
    }
    if enroll_active || wizard.is_some() || menu.wizard_open {
        return RadialAction::None;
    }
    if let Some(id) = hit_icon(mx, my, &centers) {
        let act = match id {
            IconId::Settings => RadialAction::ToggleSettings,
            IconId::Dismiss => RadialAction::Dismiss,
            IconId::Mute => RadialAction::Mute,
            IconId::Info => RadialAction::Info,
        };
        apply_action(menu, settings, act, now_ms);
        return act;
    }
    RadialAction::None
}

/// Union AABB of buddy + icons + optional panel, for paint and input shape.
pub fn overlay_bounds(
    cx: f32,
    cy: f32,
    buddy_pad: f32,
    listening: bool,
    menu: &RadialMenu,
    now_ms: u64,
    screen_w: f32,
    screen_h: f32,
) -> (f32, f32, f32, f32) {
    if menu.calib_open {
        return (0.0, 0.0, screen_w.max(1.0), screen_h.max(1.0));
    }
    let mut x0 = cx - buddy_pad;
    let mut y0 = cy - buddy_pad;
    let mut x1 = cx + buddy_pad;
    let mut y1 = cy + buddy_pad;
    if listening {
        let centers = icon_centers(cx, cy, menu.layout_ms(now_ms), true);
        for &(_, x, y) in &centers {
            x0 = x0.min(x - ICON_RADIUS - 3.0);
            y0 = y0.min(y - ICON_RADIUS - 3.0);
            x1 = x1.max(x + ICON_RADIUS + 3.0);
            y1 = y1.max(y + ICON_RADIUS + 3.0);
        }
        if menu.panel_open() {
            if let Some((_, sx, sy)) = centers
                .iter()
                .copied()
                .find(|(id, _, _)| *id == IconId::Settings)
            {
                let (px, py, pw, ph) = panel_rect(sx, sy, screen_w, screen_h);
                x0 = x0.min(px);
                y0 = y0.min(py);
                x1 = x1.max(px + pw);
                y1 = y1.max(py + ph);
            }
        }
    }
    (
        x0.clamp(0.0, screen_w),
        y0.clamp(0.0, screen_h),
        (x1 - x0).clamp(1.0, screen_w),
        (y1 - y0).clamp(1.0, screen_h),
    )
}

pub fn hit_rects(
    cx: f32,
    cy: f32,
    buddy_pad: f32,
    listening: bool,
    menu: &RadialMenu,
    now_ms: u64,
    screen_w: f32,
    screen_h: f32,
) -> Vec<(i16, i16, u16, u16)> {
    let mut r = vec![circ_rect(cx, cy, buddy_pad)];
    if menu.calib_open {
        let ((sx, sy), (cx_, cy_)) = calib_chip_centers(cx, cy);
        r.push(circ_rect(sx, sy, ICON_RADIUS + 2.0));
        r.push(circ_rect(cx_, cy_, ICON_RADIUS + 2.0));
        return r;
    }
    if listening {
        for &(_, x, y) in &icon_centers(cx, cy, menu.layout_ms(now_ms), true) {
            r.push(circ_rect(x, y, ICON_RADIUS + 2.0));
        }
        if menu.panel_open() {
            let centers = icon_centers(cx, cy, menu.layout_ms(now_ms), true);
            if let Some((_, sx, sy)) = centers
                .iter()
                .copied()
                .find(|(id, _, _)| *id == IconId::Settings)
            {
                let (px, py, pw, ph) = panel_rect(sx, sy, screen_w, screen_h);
                r.push((px as i16, py as i16, pw as u16, ph as u16));
            }
        }
    }
    r
}

fn circ_rect(x: f32, y: f32, rad: f32) -> (i16, i16, u16, u16) {
    (
        (x - rad).floor() as i16,
        (y - rad).floor() as i16,
        (rad * 2.0).ceil() as u16,
        (rad * 2.0).ceil() as u16,
    )
}

pub fn config_dir() -> Option<PathBuf> {
    crate::identity::config_dir()
}

pub fn settings_path() -> Option<PathBuf> {
    Some(config_dir()?.join("settings.ini"))
}

pub fn load_settings() -> Settings {
    let mut s = Settings::default();
    if let Some(p) = settings_path() {
        if let Ok(txt) = fs::read_to_string(p) {
            for line in txt.lines() {
                let Some((k, v)) = line.split_once('=') else {
                    continue;
                };
                match k.trim() {
                    "listen_ms" => {
                        if let Ok(n) = v.trim().parse() {
                            s.listen_ms = n;
                        }
                    }
                    "gaze_lock_ms" => {
                        if let Ok(n) = v.trim().parse() {
                            s.gaze_lock_ms = n;
                        }
                    }
                    "camera" => s.camera = v.trim() == "1" || v.trim() == "true",
                    "stroke" => {
                        if let Ok(n) = v.trim().parse() {
                            s.stroke = n;
                        }
                    }
                    "gate" => s.gate = GateMode::parse(v),
                    "gestures_need_face" => {
                        s.gestures_need_face =
                            v.trim() == "1" || v.trim().eq_ignore_ascii_case("true")
                    }
                    _ => {}
                }
            }
        }
    }
    s.clamp();
    s
}

pub fn save_settings(s: &Settings) {
    let Some(dir) = config_dir() else {
        return;
    };
    let _ = fs::create_dir_all(&dir);
    if let Some(p) = settings_path() {
        let txt = format!(
            "listen_ms={}\ngaze_lock_ms={}\ncamera={}\nstroke={:.2}\n{}",
            s.listen_ms,
            s.gaze_lock_ms,
            if s.camera { 1 } else { 0 },
            s.stroke,
            identity_ini_lines(&IdentitySettings {
                gate: s.gate,
                gestures_need_face: s.gestures_need_face,
                ..IdentitySettings::default()
            })
        );
        let _ = fs::write(p, txt);
    }
}

pub fn parse_settings_ini(txt: &str) -> Settings {
    let mut s = Settings::default();
    for line in txt.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k.trim() {
            "listen_ms" => {
                if let Ok(n) = v.trim().parse() {
                    s.listen_ms = n;
                }
            }
            "gaze_lock_ms" => {
                if let Ok(n) = v.trim().parse() {
                    s.gaze_lock_ms = n;
                }
            }
            "camera" => s.camera = v.trim() == "1" || v.trim() == "true",
            "stroke" => {
                if let Ok(n) = v.trim().parse() {
                    s.stroke = n;
                }
            }
            "gate" => s.gate = GateMode::parse(v),
            "gestures_need_face" => {
                s.gestures_need_face = v.trim() == "1" || v.trim().eq_ignore_ascii_case("true")
            }
            _ => {}
        }
    }
    let mut id = IdentitySettings::default();
    parse_identity_ini(txt, &mut id);
    s.gate = id.gate;
    s.gestures_need_face = id.gestures_need_face;
    s.clamp();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_icons_around_center() {
        let c = icon_centers(200.0, 200.0, 0, false);
        assert_eq!(c.len(), 4);
        assert_eq!(c[0].0, IconId::Settings);
        for &(_, x, y) in &c {
            let d = (x - 200.0).hypot(y - 200.0);
            assert!((d - ORBIT_RADIUS).abs() < 0.6, "{d}");
        }
        // Settings starts to the right (angle 0).
        assert!(c[0].1 > 200.0);
        assert!((c[0].2 - 200.0).abs() < 1.0);
    }

    #[test]
    fn icon_hit_radius() {
        let c = icon_centers(0.0, 0.0, 0, false);
        let (id, x, y) = c[0];
        assert_eq!(hit_icon(x, y, &c), Some(id));
        assert_eq!(hit_icon(x + ICON_RADIUS - 1.0, y, &c), Some(id));
        assert_eq!(hit_icon(400.0, 400.0, &c), None);
    }

    #[test]
    fn settings_toggle_and_close() {
        let mut menu = RadialMenu::new();
        let mut s = Settings::default();
        let centers = icon_centers(400.0, 400.0, 0, false);
        let (_, sx, sy) = centers[0];
        let act = click_listening(&mut menu, &mut s, sx, sy, 400.0, 400.0, 10, 1920.0, 1200.0);
        assert_eq!(act, RadialAction::ToggleSettings);
        assert!(menu.settings_open);
        let act = click_listening(&mut menu, &mut s, sx, sy, 400.0, 400.0, 20, 1920.0, 1200.0);
        assert_eq!(act, RadialAction::ToggleSettings);
        assert!(!menu.settings_open);
    }

    #[test]
    fn dismiss_resets_menu() {
        let mut menu = RadialMenu::new();
        menu.settings_open = true;
        let mut s = Settings::default();
        let centers = icon_centers(400.0, 400.0, 0, false);
        let (_, dx, dy) = centers
            .iter()
            .copied()
            .find(|(id, _, _)| *id == IconId::Dismiss)
            .unwrap();
        let act = click_listening(&mut menu, &mut s, dx, dy, 400.0, 400.0, 1, 1920.0, 1200.0);
        assert_eq!(act, RadialAction::Dismiss);
        assert!(!menu.settings_open);
    }

    #[test]
    fn panel_close_and_camera_toggle() {
        let mut menu = RadialMenu::new();
        menu.settings_open = true;
        let mut s = Settings::default();
        assert!(s.camera);
        let centers = icon_centers(400.0, 400.0, 0, false);
        let (_, sx, sy) = centers[0];
        let (px, py, _, _) = panel_rect(sx, sy, 1920.0, 1200.0);
        let cam = camera_hit(px, py);
        let act = hit_panel(cam.0 + 4.0, cam.1 + 4.0, px, py, &s);
        assert_eq!(act, RadialAction::ToggleCamera);
        apply_action(&mut menu, &mut s, act, 0);
        assert!(!s.camera);
        let (cx, cy) = close_button(px, py);
        assert_eq!(hit_panel(cx, cy, px, py, &s), RadialAction::CloseSettings);
    }

    #[test]
    fn sliders_change_settings() {
        let mut menu = RadialMenu::new();
        let mut s = Settings::default();
        let centers = icon_centers(400.0, 400.0, 0, false);
        let (px, py, _, _) = panel_rect(centers[0].1, centers[0].2, 1920.0, 1200.0);
        let listen = slider_track(px, py, 0);
        let act = hit_panel(listen.0 + listen.2, listen.1 + 4.0, px, py, &s);
        apply_action(&mut menu, &mut s, act, 0);
        assert!(s.listen_ms > 12_000, "{}", s.listen_ms);
    }

    #[test]
    fn bounds_grow_for_icons_and_panel() {
        let menu = RadialMenu::new();
        let (x, y, w, h) = overlay_bounds(200.0, 200.0, 100.0, false, &menu, 0, 800.0, 600.0);
        assert!((w - 200.0).abs() < 2.0);
        let listening = overlay_bounds(200.0, 200.0, 100.0, true, &menu, 0, 800.0, 600.0);
        assert!(listening.2 > w + 20.0);
        let mut open = RadialMenu::new();
        open.settings_open = true;
        let with_panel = overlay_bounds(200.0, 200.0, 100.0, true, &open, 0, 800.0, 600.0);
        assert!(with_panel.2 * with_panel.3 > listening.2 * listening.3);
        let _ = (x, y, h);
    }

    #[test]
    fn settings_ini_roundtrip() {
        let s = parse_settings_ini(
            "listen_ms=4000\ngaze_lock_ms=6000\ncamera=0\nstroke=2.2\ngate=face\ngestures_need_face=0\n",
        );
        assert_eq!(s.listen_ms, 4000);
        assert_eq!(s.gaze_lock_ms, 6000);
        assert!(!s.camera);
        assert!((s.stroke - 2.2).abs() < 0.01);
        assert_eq!(s.gate, GateMode::Face);
        assert!(!s.gestures_need_face);
    }

    #[test]
    fn hit_rects_include_icons_when_listening() {
        let menu = RadialMenu::new();
        let idle = hit_rects(100.0, 100.0, 80.0, false, &menu, 0, 800.0, 600.0);
        assert_eq!(idle.len(), 1);
        let listen = hit_rects(100.0, 100.0, 80.0, true, &menu, 0, 800.0, 600.0);
        assert_eq!(listen.len(), 1 + ICON_COUNT);
    }

    #[test]
    fn people_tab_and_enroll_hits() {
        let mut s = Settings::default();
        s.page = SettingsPage::People;
        let (px, py, _, _) = panel_rect(400.0, 400.0, 1920.0, 1200.0);
        let look = tab_look_hit(px, py);
        assert_eq!(
            hit_panel_ex(look.0 + 2.0, look.1 + 2.0, px, py, &s, 1, false, None),
            RadialAction::TabLook
        );
        let face = id_btn(px, py, 0);
        assert_eq!(
            hit_panel_ex(face.0 + 2.0, face.1 + 2.0, px, py, &s, 1, false, None),
            RadialAction::StartEnrollFace
        );
        let c = cancel_enroll_hit(px, py);
        assert_eq!(
            hit_panel_ex(c.0 + 2.0, c.1 + 2.0, px, py, &s, 1, true, None),
            RadialAction::CancelEnroll
        );
        let add = new_btn(px, py);
        assert_eq!(
            hit_panel_ex(add.0 + 2.0, add.1 + 2.0, px, py, &s, 0, false, None),
            RadialAction::OpenWizard
        );
    }

    #[test]
    fn wizard_hits_advance_skip_cancel() {
        let s = Settings::default();
        let (px, py, _, _) = panel_rect(400.0, 400.0, 1920.0, 1200.0);
        let hits = WizardHits {
            can_skip: true,
            can_next: true,
            can_add: true,
        };
        let next = wizard_next_hit(px, py);
        assert_eq!(
            hit_panel_ex(next.0 + 2.0, next.1 + 2.0, px, py, &s, 0, false, Some(hits)),
            RadialAction::WizardAdvance
        );
        let skip = wizard_skip_hit(px, py);
        assert_eq!(
            hit_panel_ex(skip.0 + 2.0, skip.1 + 2.0, px, py, &s, 0, false, Some(hits)),
            RadialAction::WizardSkip
        );
        let add = wizard_add_hit(px, py);
        assert_eq!(
            hit_panel_ex(add.0 + 2.0, add.1 + 2.0, px, py, &s, 0, false, Some(hits)),
            RadialAction::WizardAddAnother
        );
        let (cx, cy) = close_button(px, py);
        assert_eq!(
            hit_panel_ex(cx, cy, px, py, &s, 0, false, Some(hits)),
            RadialAction::WizardCancel
        );
    }

    #[test]
    fn calib_row_under_sliders_and_panel_open() {
        let s = Settings::default();
        let (px, py, _, _) = panel_rect(400.0, 400.0, 1920.0, 1200.0);
        let c = calib_hit(px, py);
        let stroke = slider_track(px, py, 2);
        assert!(c.1 > stroke.1 + stroke.3);
        assert_eq!(
            hit_panel_ex(c.0 + 2.0, c.1 + 2.0, px, py, &s, 0, false, None),
            RadialAction::StartGazeCalib
        );
        let mut menu = RadialMenu::new();
        assert!(!menu.panel_open());
        menu.calib_open = true;
        assert!(menu.panel_open());
        let bounds = overlay_bounds(200.0, 200.0, 100.0, true, &menu, 0, 800.0, 600.0);
        assert!((bounds.2 - 800.0).abs() < 1.0 && (bounds.3 - 600.0).abs() < 1.0);
        let hits = hit_rects(200.0, 200.0, 100.0, true, &menu, 0, 800.0, 600.0);
        assert_eq!(hits.len(), 3);
        let ((sx, sy), _) = calib_chip_centers(200.0, 200.0);
        assert_eq!(
            click_listening_ex(
                &mut menu,
                &mut Settings::default(),
                sx,
                sy,
                200.0,
                200.0,
                0,
                800.0,
                600.0,
                0,
                false,
                None
            ),
            RadialAction::SkipGazeCalib
        );
    }
}
