//! Native Wayland overlay via wlr-layer-shell (Hyprland / wlroots).
//!
//! Fullscreen (output-sized) transparent layer on the **overlay** stack,
//! anchored to all edges, exclusive zone 0. Click-through uses
//! `wl_surface.set_input_region` — empty except buddy / icon / panel disks.
//!
//! Multi-output: compositor-default (usually the focused / primary) output.
//! Extra heads are a follow-up.

use super::FrameInput;
use smithay_client_toolkit::compositor::{CompositorHandler, CompositorState};
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::seat::keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers};
use smithay_client_toolkit::seat::pointer::{PointerEvent, PointerEventKind, PointerHandler};
use smithay_client_toolkit::seat::{Capability, SeatHandler, SeatState};
use smithay_client_toolkit::shell::wlr_layer::{
    Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
    LayerSurfaceConfigure,
};
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shm::slot::{Buffer, SlotPool};
use smithay_client_toolkit::shm::{Shm, ShmHandler};
use smithay_client_toolkit::{
    delegate_compositor, delegate_keyboard, delegate_layer, delegate_output, delegate_pointer,
    delegate_registry, delegate_seat, delegate_shm, registry_handlers,
};
use std::os::fd::{AsFd, AsRawFd};
use std::time::{Duration, Instant};
use wayland_client::globals::registry_queue_init;
use wayland_client::protocol::{
    wl_keyboard, wl_output, wl_pointer, wl_region, wl_seat, wl_shm, wl_surface,
};
use wayland_client::{Connection, EventQueue, QueueHandle};

wayland_client::delegate_noop!(WaylandApp: ignore wl_region::WlRegion);

/// Linux evdev / Wayland BTN_LEFT.
pub const BTN_LEFT: u32 = 0x110;

pub struct WaylandDisplay {
    conn: Connection,
    event_queue: EventQueue<WaylandApp>,
    app: WaylandApp,
}

struct WaylandApp {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    compositor: CompositorState,
    shm: Shm,
    layer: LayerSurface,
    pool: SlotPool,
    /// Two shm slots; draw only after `wl_buffer.release`.
    buffers: [Option<Buffer>; 2],
    /// Last origin blit into each slot (`None` = uninitialized, full fill).
    buf_rect: [Option<(i16, i16, u16, u16)>; 2],
    /// Last origin committed to the compositor.
    prev_rect: Option<(i16, i16, u16, u16)>,
    width: u32,
    height: u32,
    configured: bool,
    closed: bool,
    hidden: bool,
    pointer_x: f32,
    pointer_y: f32,
    pointer_over: bool,
    button: bool,
    ev_press: bool,
    ev_release: bool,
    escape: bool,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    pointer: Option<wl_pointer::WlPointer>,
}

impl WaylandDisplay {
    pub fn open() -> Result<Self, Box<dyn std::error::Error>> {
        let conn =
            Connection::connect_to_env().map_err(|e| format!("wayland connect failed: {e}"))?;
        let (globals, mut event_queue) = registry_queue_init::<WaylandApp>(&conn)
            .map_err(|e| format!("wayland registry: {e}"))?;
        let qh = event_queue.handle();

        let compositor = CompositorState::bind(&globals, &qh)
            .map_err(|e| format!("wl_compositor missing: {e}"))?;
        let layer_shell = LayerShell::bind(&globals, &qh).map_err(|e| {
            format!("zwlr_layer_shell_v1 missing ({e}) — need a wlroots/Hyprland compositor")
        })?;
        let shm = Shm::bind(&globals, &qh).map_err(|e| format!("wl_shm missing: {e}"))?;

        let surface = compositor.create_surface(&qh);
        let layer =
            layer_shell.create_layer_surface(&qh, surface, Layer::Overlay, Some("buckyboi"), None);
        layer.set_anchor(Anchor::TOP | Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
        layer.set_exclusive_zone(0);
        layer.set_keyboard_interactivity(KeyboardInteractivity::OnDemand);
        layer.set_size(0, 0);
        layer.commit();

        let pool =
            SlotPool::new(1920 * 1080 * 4 * 2, &shm).map_err(|e| format!("wl_shm pool: {e}"))?;

        let mut app = WaylandApp {
            registry_state: RegistryState::new(&globals),
            seat_state: SeatState::new(&globals, &qh),
            output_state: OutputState::new(&globals, &qh),
            compositor,
            shm,
            layer,
            pool,
            buffers: [None, None],
            buf_rect: [None, None],
            prev_rect: None,
            width: 0,
            height: 0,
            configured: false,
            closed: false,
            hidden: false,
            pointer_x: 0.0,
            pointer_y: 0.0,
            pointer_over: false,
            button: false,
            ev_press: false,
            ev_release: false,
            escape: false,
            keyboard: None,
            pointer: None,
        };

        let start = Instant::now();
        while !app.configured {
            if start.elapsed() > Duration::from_secs(3) {
                return Err(
                    "wayland: layer-shell configure timeout (compositor did not size the overlay)"
                        .into(),
                );
            }
            if app.closed {
                return Err("wayland: compositor closed the layer surface".into());
            }
            pump(&conn, &mut event_queue, &mut app, 50)?;
        }

        if app.width == 0 || app.height == 0 {
            app.width = 1920;
            app.height = 1080;
            eprintln!("buckyboi: layer configure had 0×0 — assuming 1920×1080");
        }

        // Map a fully click-through transparent buffer so the surface exists.
        commit_clear(&mut app)?;
        conn.flush()?;

        eprintln!(
            "buckyboi: layer-shell overlay {}×{} (namespace buckyboi, exclusive_zone=0)",
            app.width, app.height
        );

        Ok(Self {
            conn,
            event_queue,
            app,
        })
    }

    pub fn size(&self) -> (u16, u16) {
        (
            self.app.width.min(u16::MAX as u32) as u16,
            self.app.height.min(u16::MAX as u32) as u16,
        )
    }

    pub fn poll_input(&mut self) -> Result<FrameInput, Box<dyn std::error::Error>> {
        pump(&self.conn, &mut self.event_queue, &mut self.app, 0)?;
        let ev_press = std::mem::take(&mut self.app.ev_press);
        let ev_release = std::mem::take(&mut self.app.ev_release);
        let escape = std::mem::take(&mut self.app.escape);
        Ok(FrameInput {
            mx: self.app.pointer_x,
            my: self.app.pointer_y,
            button: self.app.button || (ev_press && !ev_release),
            ev_press,
            ev_release,
            ev_xy: self
                .app
                .pointer_over
                .then_some((self.app.pointer_x, self.app.pointer_y)),
            escape,
            wake: false,
            quit: self.app.closed,
        })
    }

    pub fn present(
        &mut self,
        pixels: &[u8],
        origin: (i16, i16, u16, u16),
        hits: &[(i16, i16, u16, u16)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        pump(&self.conn, &mut self.event_queue, &mut self.app, 0)?;
        self.app.hidden = false;
        if !self.app.configured {
            // After an unmap (`attach(None)`), wlroots rejects the next buffer
            // until a fresh layer-surface configure. Commit without a buffer
            // and wait for `configure` on a later frame.
            self.app.layer.commit();
            self.conn.flush()?;
            return Ok(());
        }
        let qh = self.event_queue.handle();
        apply_hits(&self.app, &qh, hits);
        commit_pixels(&mut self.app, pixels, origin)?;
        self.conn.flush()?;
        Ok(())
    }

    pub fn hide(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        pump(&self.conn, &mut self.event_queue, &mut self.app, 0)?;
        self.app.hidden = true;
        let qh = self.event_queue.handle();
        apply_hits(&self.app, &qh, &[]);
        // Prefer a transparent buffer (still mapped). `attach(None)` unmaps the
        // layer surface and Hyprland then requires a new configure before the
        // next buffer — that is the wake crash.
        if commit_clear(&mut self.app)? {
            self.conn.flush()?;
            return Ok(());
        }
        self.app.layer.wl_surface().attach(None, 0, 0);
        self.app.layer.commit();
        self.app.configured = false;
        drop_slots(&mut self.app);
        self.conn.flush()?;
        Ok(())
    }

    pub fn show(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.app.hidden = false;
        Ok(())
    }

    pub fn raise(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Overlay layer is already above windows; nothing to restack.
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.flush()?;
        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let qh = self.event_queue.handle();
        apply_hits(&self.app, &qh, &[]);
        self.app.layer.wl_surface().attach(None, 0, 0);
        self.app.layer.commit();
        self.conn.flush()?;
        Ok(())
    }
}

fn pump(
    conn: &Connection,
    queue: &mut EventQueue<WaylandApp>,
    app: &mut WaylandApp,
    timeout_ms: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    queue.dispatch_pending(app)?;
    conn.flush()?;
    match queue.prepare_read() {
        Some(guard) => {
            let mut pfd = libc::pollfd {
                fd: conn.as_fd().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let n = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
            if n > 0 {
                let _ = guard.read();
                queue.dispatch_pending(app)?;
            } else {
                drop(guard);
            }
        }
        None => {
            queue.dispatch_pending(app)?;
        }
    }
    conn.flush()?;
    Ok(())
}

fn apply_hits(app: &WaylandApp, qh: &QueueHandle<WaylandApp>, hits: &[(i16, i16, u16, u16)]) {
    let region = app.compositor.wl_compositor().create_region(qh, ());
    for &(x, y, w, h) in hits {
        if w == 0 || h == 0 {
            continue;
        }
        region.add(x as i32, y as i32, w as i32, h as i32);
    }
    app.layer.wl_surface().set_input_region(Some(&region));
    region.destroy();
}

const DAMAGE_PAD: i32 = 2;

fn drop_slots(app: &mut WaylandApp) {
    app.buffers = [None, None];
    app.buf_rect = [None, None];
    app.prev_rect = None;
}

/// Released slot, or an empty slot to create. `None` if both are outstanding.
fn pick_released(app: &mut WaylandApp) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let w = app.width.max(1);
    let h = app.height.max(1);
    let stride = (w as i32) * 4;
    let height = h as i32;

    let stale = app.buffers.iter().any(|slot| {
        slot.as_ref()
            .is_some_and(|b| b.height() != height || b.stride() != stride)
    });
    if stale {
        drop_slots(app);
    }

    let mut idx = None;
    {
        let pool = &mut app.pool;
        for (i, slot) in app.buffers.iter().enumerate() {
            if let Some(buf) = slot {
                if buf.canvas(pool).is_some() {
                    idx = Some(i);
                    break;
                }
            }
        }
    }
    if idx.is_none() {
        idx = app.buffers.iter().position(Option::is_none);
    }
    let Some(i) = idx else {
        return Ok(None);
    };
    if app.buffers[i].is_some() {
        return Ok(Some(i));
    }

    match app
        .pool
        .create_buffer(w as i32, height, stride, wl_shm::Format::Argb8888)
    {
        Ok((buffer, _)) => {
            app.buffers[i] = Some(buffer);
            app.buf_rect[i] = None;
            Ok(Some(i))
        }
        Err(_) => {
            drop_slots(app);
            let need = (w as usize) * (h as usize) * 4;
            app.pool = SlotPool::new((need * 2).max(4096), &app.shm)
                .map_err(|e| format!("wl_shm resize: {e}"))?;
            let (buffer, _) = app
                .pool
                .create_buffer(w as i32, height, stride, wl_shm::Format::Argb8888)
                .map_err(|e| format!("wl_shm buffer: {e}"))?;
            app.buffers[i] = Some(buffer);
            app.buf_rect[i] = None;
            Ok(Some(i))
        }
    }
}

fn paint_slot(
    app: &mut WaylandApp,
    i: usize,
    leftover: Option<(i16, i16, u16, u16)>,
    origin: (i16, i16, u16, u16),
    pixels: Option<&[u8]>,
) -> bool {
    let w = app.width.max(1);
    let h = app.height.max(1);
    let canvas = match app.buffers[i].as_ref() {
        Some(buf) => buf.canvas(&mut app.pool),
        None => None,
    };
    let Some(canvas) = canvas else {
        return false;
    };
    match leftover {
        None => canvas.fill(0),
        Some(old) => zero_rect(canvas, w, h, old),
    }
    if let Some(px) = pixels {
        blit_argb(canvas, w, h, origin, px);
    }
    true
}

fn attach_slot(app: &mut WaylandApp, i: usize) -> bool {
    let surface = app.layer.wl_surface();
    let Some(buf) = app.buffers[i].as_ref() else {
        return false;
    };
    buf.attach_to(surface).is_ok()
}

fn commit_clear(app: &mut WaylandApp) -> Result<bool, Box<dyn std::error::Error>> {
    let w = app.width.max(1);
    let h = app.height.max(1);
    let Some(i) = pick_released(app)? else {
        return Ok(false);
    };
    if !paint_slot(app, i, None, (0, 0, 0, 0), None) {
        return Ok(false);
    }
    if !attach_slot(app, i) {
        app.buf_rect[i] = None;
        return Ok(false);
    }
    app.layer
        .wl_surface()
        .damage_buffer(0, 0, w as i32, h as i32);
    app.layer.commit();
    app.buf_rect[i] = Some((0, 0, 0, 0));
    app.prev_rect = None;
    Ok(true)
}

fn commit_pixels(
    app: &mut WaylandApp,
    pixels: &[u8],
    origin: (i16, i16, u16, u16),
) -> Result<(), Box<dyn std::error::Error>> {
    let w = app.width.max(1);
    let h = app.height.max(1);
    let Some(i) = pick_released(app)? else {
        // Never block present on wl_buffer.release.
        return Ok(());
    };
    let leftover = app.buf_rect[i];
    let prev = app.prev_rect;
    if !paint_slot(app, i, leftover, origin, Some(pixels)) {
        return Ok(());
    }
    if !attach_slot(app, i) {
        app.buf_rect[i] = None;
        return Ok(());
    }
    if leftover.is_none() {
        app.layer
            .wl_surface()
            .damage_buffer(0, 0, w as i32, h as i32);
    } else {
        let dmg = match prev {
            Some(p) => union_rect(p, origin),
            None => origin,
        };
        let (dx, dy, dw, dh) = pad_rect(dmg, w, h);
        if dw > 0 && dh > 0 {
            app.layer.wl_surface().damage_buffer(dx, dy, dw, dh);
        }
    }
    app.layer.commit();
    app.buf_rect[i] = Some(origin);
    app.prev_rect = Some(origin);
    Ok(())
}

fn zero_rect(dst: &mut [u8], dw: u32, dh: u32, rect: (i16, i16, u16, u16)) {
    let (ox, oy, rw, rh) = rect;
    if rw == 0 || rh == 0 {
        return;
    }
    let dw_i = dw as i32;
    let dh_i = dh as i32;
    for row in 0..rh as i32 {
        let dy = oy as i32 + row;
        if dy < 0 || dy >= dh_i {
            continue;
        }
        let mut dx = ox as i32;
        let mut copy_w = rw as i32;
        if dx < 0 {
            copy_w += dx;
            dx = 0;
        }
        if dx + copy_w > dw_i {
            copy_w = dw_i - dx;
        }
        if copy_w <= 0 {
            continue;
        }
        let d = (dy as usize * dw as usize + dx as usize) * 4;
        let n = copy_w as usize * 4;
        if d + n <= dst.len() {
            dst[d..d + n].fill(0);
        }
    }
}

fn union_rect(a: (i16, i16, u16, u16), b: (i16, i16, u16, u16)) -> (i16, i16, u16, u16) {
    if a.2 == 0 || a.3 == 0 {
        return b;
    }
    if b.2 == 0 || b.3 == 0 {
        return a;
    }
    let x0 = a.0.min(b.0);
    let y0 = a.1.min(b.1);
    let x1 = (i32::from(a.0) + i32::from(a.2)).max(i32::from(b.0) + i32::from(b.2));
    let y1 = (i32::from(a.1) + i32::from(a.3)).max(i32::from(b.1) + i32::from(b.3));
    let w = (x1 - i32::from(x0)).clamp(0, i32::from(u16::MAX)) as u16;
    let h = (y1 - i32::from(y0)).clamp(0, i32::from(u16::MAX)) as u16;
    (x0, y0, w, h)
}

fn pad_rect(r: (i16, i16, u16, u16), dw: u32, dh: u32) -> (i32, i32, i32, i32) {
    if r.2 == 0 || r.3 == 0 {
        return (0, 0, 0, 0);
    }
    let x0 = (i32::from(r.0) - DAMAGE_PAD).max(0);
    let y0 = (i32::from(r.1) - DAMAGE_PAD).max(0);
    let x1 = (i32::from(r.0) + i32::from(r.2) + DAMAGE_PAD).min(dw as i32);
    let y1 = (i32::from(r.1) + i32::from(r.3) + DAMAGE_PAD).min(dh as i32);
    (x0, y0, (x1 - x0).max(0), (y1 - y0).max(0))
}

fn blit_argb(dst: &mut [u8], dw: u32, dh: u32, origin: (i16, i16, u16, u16), src: &[u8]) {
    let (ox, oy, sw, sh) = origin;
    let dw_i = dw as i32;
    let dh_i = dh as i32;
    for row in 0..sh as i32 {
        let dy = oy as i32 + row;
        if dy < 0 || dy >= dh_i {
            continue;
        }
        let mut sx = 0i32;
        let mut dx = ox as i32;
        if dx < 0 {
            sx = -dx;
            dx = 0;
        }
        let mut copy_w = sw as i32 - sx;
        if dx + copy_w > dw_i {
            copy_w = dw_i - dx;
        }
        if copy_w <= 0 {
            continue;
        }
        let s = (row as usize * sw as usize + sx as usize) * 4;
        let d = (dy as usize * dw as usize + dx as usize) * 4;
        let n = copy_w as usize * 4;
        if s + n <= src.len() && d + n <= dst.len() {
            dst[d..d + n].copy_from_slice(&src[s..s + n]);
        }
    }
}

impl CompositorHandler for WaylandApp {
    fn scale_factor_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_factor: i32,
    ) {
    }

    fn transform_changed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _new_transform: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _time: u32,
    ) {
    }

    fn surface_enter(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _surface: &wl_surface::WlSurface,
        _output: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for WaylandApp {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        if let Some(info) = self.output_state.info(&output) {
            eprintln!(
                "buckyboi: wayland output {:?} loc={:?} scale={}",
                info.name, info.location, info.scale_factor
            );
        }
    }

    fn update_output(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }

    fn output_destroyed(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _output: wl_output::WlOutput,
    ) {
    }
}

impl LayerShellHandler for WaylandApp {
    fn closed(&mut self, _conn: &Connection, _qh: &QueueHandle<Self>, _layer: &LayerSurface) {
        self.closed = true;
    }

    fn configure(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let mut resized = false;
        if configure.new_size.0 > 0 {
            resized |= self.width != configure.new_size.0;
            self.width = configure.new_size.0;
        }
        if configure.new_size.1 > 0 {
            resized |= self.height != configure.new_size.1;
            self.height = configure.new_size.1;
        }
        self.configured = true;
        self.buf_rect = [None, None];
        self.prev_rect = None;
        if resized {
            self.buffers = [None, None];
        }
    }
}

impl SeatHandler for WaylandApp {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}

    fn new_capability(
        &mut self,
        _conn: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Keyboard && self.keyboard.is_none() {
            match self.seat_state.get_keyboard(qh, &seat, None) {
                Ok(k) => self.keyboard = Some(k),
                Err(e) => eprintln!(
                    "buckyboi: wl_keyboard unavailable ({e}) — quit with SUPER+Escape or `buckyboi --quit`"
                ),
            }
        }
        if capability == Capability::Pointer && self.pointer.is_none() {
            match self.seat_state.get_pointer(qh, &seat) {
                Ok(p) => self.pointer = Some(p),
                Err(e) => eprintln!("buckyboi: wl_pointer unavailable ({e})"),
            }
        }
    }

    fn remove_capability(
        &mut self,
        _conn: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Keyboard {
            if let Some(k) = self.keyboard.take() {
                k.release();
            }
        }
        if capability == Capability::Pointer {
            if let Some(p) = self.pointer.take() {
                p.release();
            }
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl KeyboardHandler for WaylandApp {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        surface: &wl_surface::WlSurface,
        _: u32,
        _: &[u32],
        _: &[Keysym],
    ) {
        let _ = surface;
    }

    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
    ) {
    }

    fn press_key(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        if event.keysym == Keysym::Escape {
            self.escape = true;
        }
    }

    fn release_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: KeyEvent,
    ) {
    }

    fn update_modifiers(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: Modifiers,
        _: u32,
    ) {
    }
}

impl PointerHandler for WaylandApp {
    fn pointer_frame(
        &mut self,
        _conn: &Connection,
        _qh: &QueueHandle<Self>,
        _pointer: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            if &event.surface != self.layer.wl_surface() {
                continue;
            }
            self.pointer_x = event.position.0 as f32;
            self.pointer_y = event.position.1 as f32;
            match event.kind {
                PointerEventKind::Enter { .. } => self.pointer_over = true,
                PointerEventKind::Leave { .. } => self.pointer_over = false,
                PointerEventKind::Motion { .. } => {}
                PointerEventKind::Press { button, .. } if button == BTN_LEFT => {
                    self.button = true;
                    self.ev_press = true;
                }
                PointerEventKind::Release { button, .. } if button == BTN_LEFT => {
                    self.button = false;
                    self.ev_release = true;
                }
                _ => {}
            }
        }
    }
}

impl ShmHandler for WaylandApp {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

delegate_compositor!(WaylandApp);
delegate_output!(WaylandApp);
delegate_shm!(WaylandApp);
delegate_seat!(WaylandApp);
delegate_keyboard!(WaylandApp);
delegate_pointer!(WaylandApp);
delegate_layer!(WaylandApp);
delegate_registry!(WaylandApp);

impl ProvidesRegistryState for WaylandApp {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn left_button_is_linux_btn_left() {
        assert_eq!(BTN_LEFT, 0x110);
    }

    #[test]
    fn blit_writes_origin() {
        let mut dst = vec![0u8; 8 * 4 * 4];
        let src = vec![9u8; 2 * 2 * 4];
        blit_argb(&mut dst, 8, 4, (1, 1, 2, 2), &src);
        let i = (1 * 8 + 1) * 4;
        assert_eq!(dst[i], 9);
        assert_eq!(dst[0], 0);
    }

    #[test]
    fn zero_rect_clears_only_target() {
        let mut dst = vec![7u8; 8 * 4 * 4];
        zero_rect(&mut dst, 8, 4, (1, 1, 2, 2));
        assert_eq!(dst[0], 7);
        let i = (1 * 8 + 1) * 4;
        assert_eq!(&dst[i..i + 8], &[0u8; 8]);
        let j = (2 * 8 + 1) * 4;
        assert_eq!(&dst[j..j + 8], &[0u8; 8]);
        assert_eq!(dst[(1 * 8 + 3) * 4], 7);
    }

    #[test]
    fn zero_rect_empty_is_noop() {
        let mut dst = vec![3u8; 16];
        zero_rect(&mut dst, 2, 2, (0, 0, 0, 1));
        assert_eq!(dst, vec![3u8; 16]);
    }

    #[test]
    fn union_rect_covers_prev_and_new() {
        assert_eq!(
            union_rect((10, 10, 20, 20), (20, 15, 20, 20)),
            (10, 10, 30, 25)
        );
        assert_eq!(union_rect((4, 8, 2, 2), (10, 8, 2, 2)), (4, 8, 8, 2));
        assert_eq!(union_rect((0, 0, 0, 0), (5, 6, 7, 8)), (5, 6, 7, 8));
    }

    #[test]
    fn pad_rect_clamps_to_surface() {
        assert_eq!(pad_rect((0, 0, 4, 4), 10, 10), (0, 0, 6, 6));
        assert_eq!(pad_rect((8, 8, 2, 2), 10, 10), (6, 6, 4, 4));
        assert_eq!(pad_rect((0, 0, 0, 0), 10, 10), (0, 0, 0, 0));
    }

    #[test]
    fn wayland_open_skips_without_socket() {
        if std::env::var_os("WAYLAND_DISPLAY").is_none() {
            return;
        }
        // Nested compositor jobs may or may not advertise layer-shell.
        let _ = WaylandDisplay::open();
    }
}
