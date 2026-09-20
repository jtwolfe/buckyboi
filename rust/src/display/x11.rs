//! X11 Shape + XFixes fullscreen overlay (existing product path).

use super::FrameInput;
use std::time::Instant;
use x11rb::connection::{Connection, RequestConnection};
use x11rb::protocol::shape::{self, ConnectionExt as _};
use x11rb::protocol::xfixes::ConnectionExt as _;
use x11rb::protocol::xproto::*;
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;

pub const X11_BACKEND_NAME: &str = "x11";
const ESCAPE_KEYSYM: u32 = 0xFF1B;

pub struct X11Display {
    conn: RustConnection,
    win: Window,
    gc: Gcontext,
    root: Window,
    sw: u16,
    sh: u16,
    esc: Vec<u8>,
    mapped: bool,
    prev_rect: Option<(i16, i16, u16, u16)>,
    last_root: (i16, i16),
    have_last_root: bool,
    last_keys: [u8; 32],
    have_keys: bool,
    last_raise: Instant,
}

impl X11Display {
    pub fn open() -> Result<Self, Box<dyn std::error::Error>> {
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
        set_hit_region(&conn, win, &[])?;

        let gc = conn.generate_id()?;
        conn.create_gc(gc, win, &CreateGCAux::new())?;

        conn.map_window(win)?;
        conn.configure_window(win, &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE))?;
        conn.flush()?;

        Ok(Self {
            esc: escape_keycodes(&conn, &setup),
            conn,
            win,
            gc,
            root,
            sw,
            sh,
            mapped: true,
            prev_rect: None,
            last_root: (0, 0),
            have_last_root: false,
            last_keys: [0u8; 32],
            have_keys: false,
            last_raise: Instant::now(),
        })
    }

    pub fn size(&self) -> (u16, u16) {
        (self.sw, self.sh)
    }

    pub fn poll_input(&mut self) -> Result<FrameInput, Box<dyn std::error::Error>> {
        let mut ev_press = false;
        let mut ev_release = false;
        let mut ev_xy = None;
        while let Some(ev) = self.conn.poll_for_event()? {
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

        let ptr = self.conn.query_pointer(self.root)?.reply()?;
        let mx = ev_xy.map(|(x, _)| x).unwrap_or(ptr.root_x as f32);
        let my = ev_xy.map(|(_, y)| y).unwrap_or(ptr.root_y as f32);
        let button = ptr.mask.contains(KeyButMask::BUTTON1) || (ev_press && !ev_release);
        let keys = self.conn.query_keymap()?.reply()?.keys;

        let escape = any_escape(&keys, &self.esc);
        let moved = self.have_last_root
            && ((ptr.root_x - self.last_root.0).abs() >= 1
                || (ptr.root_y - self.last_root.1).abs() >= 1);
        let keyed = self.have_keys && keys != self.last_keys;

        self.last_root = (ptr.root_x, ptr.root_y);
        self.have_last_root = true;
        self.last_keys = keys;
        self.have_keys = true;

        Ok(FrameInput {
            mx,
            my,
            button,
            ev_press,
            ev_release,
            ev_xy,
            escape,
            wake: moved || keyed,
            quit: false,
        })
    }

    pub fn present(
        &mut self,
        pixels: &[u8],
        origin: (i16, i16, u16, u16),
        hits: &[(i16, i16, u16, u16)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.mapped {
            self.show()?;
        }
        let (ox, oy, w, h) = origin;
        if let Some(old) = self.prev_rect {
            if old != origin {
                clear_rect(&self.conn, self.win, self.gc, old.0, old.1, old.2, old.3)?;
            }
        }
        self.conn.put_image(
            ImageFormat::Z_PIXMAP,
            self.win,
            self.gc,
            w,
            h,
            ox,
            oy,
            0,
            32,
            pixels,
        )?;
        set_hit_region(&self.conn, self.win, hits)?;
        self.prev_rect = Some(origin);
        if self.last_raise.elapsed() > std::time::Duration::from_secs(2) {
            self.raise()?;
            self.last_raise = Instant::now();
        }
        Ok(())
    }

    pub fn hide(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some((x, y, w, h)) = self.prev_rect.take() {
            let _ = clear_rect(&self.conn, self.win, self.gc, x, y, w, h);
        }
        let _ = set_hit_region(&self.conn, self.win, &[]);
        self.conn.unmap_window(self.win)?;
        self.mapped = false;
        Ok(())
    }

    pub fn show(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.map_window(self.win)?;
        self.conn.configure_window(
            self.win,
            &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE),
        )?;
        self.mapped = true;
        Ok(())
    }

    pub fn raise(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.configure_window(
            self.win,
            &ConfigureWindowAux::new().stack_mode(StackMode::ABOVE),
        )?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.conn.flush()?;
        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let _ = set_hit_region(&self.conn, self.win, &[]);
        self.conn.unmap_window(self.win)?;
        self.conn.destroy_window(self.win)?;
        self.conn.flush()?;
        Ok(())
    }
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
    conn.put_image(ImageFormat::Z_PIXMAP, win, gc, w, h, x, y, 0, 32, &zeros)?;
    Ok(())
}
