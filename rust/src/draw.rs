//! Software strokes for the overlay: depth-sorted halo + core wireframe.

use crate::sim::{mix_wire, projected_edges, State};

/// Far edges first (small z), near last — nearer strokes cover farther ones.
pub fn edges_far_to_near(s: &State) -> [(f32, f32, f32, f32, f32); 30] {
    let mut e = projected_edges(s);
    e.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));
    e
}

pub fn plot(buf: &mut [u8], w: u16, h: u16, x: i32, y: i32, r: u8, g: u8, b: u8, a: u8) {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
        return;
    }
    let i = (y as usize * w as usize + x as usize) * 4;
    let da = a as u32;
    let oa = buf[i + 3] as u32;
    let out_a = da + oa.saturating_sub(da * oa / 255);
    if out_a == 0 {
        return;
    }
    let blend = |s: u8, d: u8| -> u8 {
        ((s as u32 * da + d as u32 * (255 - da) / 255 * oa / 255.max(1)) / out_a.max(1)) as u8
    };
    buf[i] = blend(b, buf[i]);
    buf[i + 1] = blend(g, buf[i + 1]);
    buf[i + 2] = blend(r, buf[i + 2]);
    buf[i + 3] = out_a as u8;
}

pub fn stamp(buf: &mut [u8], w: u16, h: u16, x: i32, y: i32, rad: i32, r: u8, g: u8, b: u8, a: u8) {
    if rad <= 0 {
        plot(buf, w, h, x, y, r, g, b, a);
        return;
    }
    let r2 = rad * rad;
    for dy in -rad..=rad {
        for dx in -rad..=rad {
            if dx * dx + dy * dy <= r2 {
                plot(buf, w, h, x + dx, y + dy, r, g, b, a);
            }
        }
    }
}

pub fn stroke(
    buf: &mut [u8],
    w: u16,
    h: u16,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    width: f32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) {
    let steps = ((x1 - x0).abs().max((y1 - y0).abs()) * 1.15) as i32 + 1;
    let steps = steps.max(1);
    let rad = (width * 0.5).round().max(0.0) as i32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x0 + (x1 - x0) * t).round() as i32;
        let y = (y0 + (y1 - y0) * t).round() as i32;
        stamp(buf, w, h, x, y, rad, r, g, b, a);
    }
}

pub fn fill_disk(
    buf: &mut [u8],
    w: u16,
    h: u16,
    cx: f32,
    cy: f32,
    radius: f32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) {
    stamp(
        buf,
        w,
        h,
        cx.round() as i32,
        cy.round() as i32,
        radius.round() as i32,
        r,
        g,
        b,
        a,
    );
}

pub fn fill_rect(
    buf: &mut [u8],
    w: u16,
    h: u16,
    x: i32,
    y: i32,
    rw: i32,
    rh: i32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) {
    for yy in y..y + rh {
        for xx in x..x + rw {
            plot(buf, w, h, xx, yy, r, g, b, a);
        }
    }
}

/// Two-pass occluding wire: dark halo, then bright core. Edges far → near.
pub fn paint_wireframe(
    buf: &mut [u8],
    w: u16,
    h: u16,
    state: &State,
    off_x: f32,
    off_y: f32,
    listening: bool,
    stroke_w: f32,
) {
    let halo_w = (stroke_w * 2.35 + 1.15).clamp(2.2, 7.0);
    let core_w = stroke_w.clamp(0.8, 3.2);
    for (ax, ay, bx, by, z) in edges_far_to_near(state) {
        let mut c = mix_wire(z);
        if listening {
            let nr = 0x7Eu32;
            let ng = 0xE8u32;
            let nb = 0xFFu32;
            let cr = (c >> 16) & 255;
            let cg = (c >> 8) & 255;
            let cb = c & 255;
            c = ((cr * 35 + nr * 65) / 100) << 16
                | ((cg * 35 + ng * 65) / 100) << 8
                | ((cb * 35 + nb * 65) / 100);
        }
        let r = ((c >> 16) & 255) as u8;
        let g = ((c >> 8) & 255) as u8;
        let b = (c & 255) as u8;
        let hr = (r as u16 * 22 / 100) as u8;
        let hg = (g as u16 * 24 / 100) as u8;
        let hb = (b as u16 * 28 / 100) as u8;
        stroke(
            buf,
            w,
            h,
            ax - off_x,
            ay - off_y,
            bx - off_x,
            by - off_y,
            halo_w,
            hr,
            hg,
            hb,
            250,
        );
        stroke(
            buf,
            w,
            h,
            ax - off_x,
            ay - off_y,
            bx - off_x,
            by - off_y,
            core_w,
            r,
            g,
            b,
            250,
        );
    }
}

/// 3×5 caps + digits for the settings panel.
fn glyph(c: char) -> [u8; 5] {
    match c {
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b111, 0b100, 0b100, 0b100, 0b111],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b111, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b111, 0b100, 0b100],
        'G' => [0b111, 0b100, 0b101, 0b101, 0b111],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b111],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'O' => [0b111, 0b101, 0b101, 0b101, 0b111],
        'P' => [0b111, 0b101, 0b111, 0b100, 0b100],
        'Q' => [0b111, 0b101, 0b101, 0b111, 0b001],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b111, 0b100, 0b111, 0b001, 0b111],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b001, 0b001, 0b001],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '+' => [0b000, 0b010, 0b111, 0b010, 0b000],
        '?' => [0b111, 0b001, 0b011, 0b000, 0b010],
        ' ' => [0b000, 0b000, 0b000, 0b000, 0b000],
        _ => [0b111, 0b101, 0b101, 0b101, 0b111],
    }
}

pub fn text(
    buf: &mut [u8],
    w: u16,
    h: u16,
    x: i32,
    y: i32,
    s: &str,
    scale: i32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) {
    let mut cx = x;
    for ch in s.chars() {
        let rows = glyph(ch.to_ascii_uppercase());
        for (row, bits) in rows.iter().enumerate() {
            for col in 0..3 {
                if bits & (1 << (2 - col)) != 0 {
                    fill_rect(
                        buf,
                        w,
                        h,
                        cx + col * scale,
                        y + row as i32 * scale,
                        scale,
                        scale,
                        r,
                        g,
                        b,
                        a,
                    );
                }
            }
        }
        cx += 4 * scale;
    }
}

use crate::radial::{
    calib_chip_centers, calib_hit, camera_hit, cancel_enroll_hit, close_button, del_btn, gate_hit,
    icon_centers, id_btn, need_face_hit, new_btn, panel_rect, person_row_hit, slider_track,
    tab_look_hit, tab_people_hit, wizard_add_hit, wizard_next_hit, wizard_skip_hit, IconId,
    IdentityHud, RadialMenu, Settings, SettingsPage, CLOSE_R, ICON_RADIUS, PANEL_H, PANEL_W,
};

pub fn paint_icon(
    buf: &mut [u8],
    w: u16,
    h: u16,
    x: f32,
    y: f32,
    id: IconId,
    hot: bool,
    flash: bool,
    off_x: f32,
    off_y: f32,
) {
    let x = x - off_x;
    let y = y - off_y;
    let (fr, fg, fb) = if flash {
        (0xF4, 0xFB, 0xFF)
    } else if hot {
        (0x9A, 0xF0, 0xFF)
    } else {
        (0x2A, 0x3A, 0x48)
    };
    fill_disk(buf, w, h, x, y, ICON_RADIUS + 1.5, 0x0C, 0x14, 0x1C, 230);
    fill_disk(buf, w, h, x, y, ICON_RADIUS, fr, fg, fb, 240);
    fill_disk(buf, w, h, x, y, ICON_RADIUS - 3.0, 0x18, 0x24, 0x30, 235);
    let (ir, ig, ib) = if flash || hot {
        (0xE8, 0xF6, 0xFF)
    } else {
        (0xC4, 0xD4, 0xE2)
    };
    match id {
        IconId::Settings => {
            for k in 0..6 {
                let a = k as f32 * std::f32::consts::TAU / 6.0;
                stroke(
                    buf,
                    w,
                    h,
                    x + 4.5 * a.cos(),
                    y + 4.5 * a.sin(),
                    x + 8.2 * a.cos(),
                    y + 8.2 * a.sin(),
                    2.4,
                    ir,
                    ig,
                    ib,
                    250,
                );
            }
            fill_disk(buf, w, h, x, y, 3.2, ir, ig, ib, 250);
            fill_disk(buf, w, h, x, y, 1.4, 0x18, 0x24, 0x30, 250);
        }
        IconId::Dismiss => {
            stroke(
                buf,
                w,
                h,
                x - 5.0,
                y - 5.0,
                x + 5.0,
                y + 5.0,
                2.0,
                ir,
                ig,
                ib,
                250,
            );
            stroke(
                buf,
                w,
                h,
                x + 5.0,
                y - 5.0,
                x - 5.0,
                y + 5.0,
                2.0,
                ir,
                ig,
                ib,
                250,
            );
        }
        IconId::Mute => {
            stroke(
                buf,
                w,
                h,
                x - 5.0,
                y - 3.0,
                x - 2.0,
                y - 3.0,
                1.8,
                ir,
                ig,
                ib,
                250,
            );
            stroke(
                buf,
                w,
                h,
                x - 2.0,
                y - 5.0,
                x + 3.0,
                y - 1.0,
                1.8,
                ir,
                ig,
                ib,
                250,
            );
            stroke(
                buf,
                w,
                h,
                x - 2.0,
                y + 5.0,
                x + 3.0,
                y + 1.0,
                1.8,
                ir,
                ig,
                ib,
                250,
            );
            stroke(
                buf,
                w,
                h,
                x - 5.0,
                y + 3.0,
                x - 2.0,
                y + 3.0,
                1.8,
                ir,
                ig,
                ib,
                250,
            );
            stroke(
                buf,
                w,
                h,
                x - 4.0,
                y + 6.0,
                x + 5.0,
                y - 6.0,
                1.6,
                0x7E,
                0xE8,
                0xFF,
                240,
            );
        }
        IconId::Info => {
            fill_disk(buf, w, h, x, y - 3.5, 1.3, ir, ig, ib, 250);
            stroke(buf, w, h, x, y - 0.5, x, y + 5.0, 1.8, ir, ig, ib, 250);
        }
    }
}

fn paint_btn(
    buf: &mut [u8],
    w: u16,
    h: u16,
    hit: (f32, f32, f32, f32),
    off_x: f32,
    off_y: f32,
    label: &str,
    on: bool,
) {
    let x = (hit.0 - off_x).round() as i32;
    let y = (hit.1 - off_y).round() as i32;
    fill_rect(
        buf,
        w,
        h,
        x,
        y,
        hit.2 as i32,
        hit.3 as i32,
        if on { 0x1E } else { 0x28 },
        if on { 0x5A } else { 0x30 },
        if on { 0x68 } else { 0x38 },
        240,
    );
    text(buf, w, h, x + 4, y + 3, label, 2, 0xE8, 0xF4, 0xFC, 250);
}

pub fn paint_panel(
    buf: &mut [u8],
    w: u16,
    h: u16,
    px: f32,
    py: f32,
    settings: &Settings,
    hud: &IdentityHud,
    off_x: f32,
    off_y: f32,
) {
    let x = (px - off_x).round() as i32;
    let y = (py - off_y).round() as i32;
    let pw = PANEL_W as i32;
    let ph = PANEL_H as i32;
    fill_rect(buf, w, h, x, y, pw, ph, 0x12, 0x18, 0x22, 245);
    fill_rect(buf, w, h, x, y, pw, 3, 0x4A, 0xC0, 0xD8, 220);
    text(
        buf,
        w,
        h,
        x + 10,
        y + 8,
        "BUCKYBOI",
        2,
        0xDE,
        0xEC,
        0xF6,
        250,
    );
    let (cx, cy) = close_button(px, py);
    fill_disk(
        buf,
        w,
        h,
        cx - off_x,
        cy - off_y,
        CLOSE_R,
        0x2A,
        0x34,
        0x40,
        240,
    );
    stroke(
        buf,
        w,
        h,
        cx - off_x - 3.5,
        cy - off_y - 3.5,
        cx - off_x + 3.5,
        cy - off_y + 3.5,
        1.6,
        0xE0,
        0xE8,
        0xF0,
        250,
    );
    stroke(
        buf,
        w,
        h,
        cx - off_x + 3.5,
        cy - off_y - 3.5,
        cx - off_x - 3.5,
        cy - off_y + 3.5,
        1.6,
        0xE0,
        0xE8,
        0xF0,
        250,
    );

    if hud.wizard_open {
        let title: String = hud.wizard_title.chars().take(16).collect();
        text(buf, w, h, x + 10, y + 32, &title, 2, 0x7E, 0xE8, 0xFF, 250);
        text(
            buf,
            w,
            h,
            x + 10,
            y + 64,
            "ENROLL",
            2,
            0xB0,
            0xC0,
            0xD0,
            240,
        );
        let body: String = hud.wizard_body.chars().take(16).collect();
        text(buf, w, h, x + 10, y + 92, &body, 2, 0x7E, 0xE8, 0xFF, 250);
        fill_rect(
            buf,
            w,
            h,
            x + 10,
            y + 130,
            pw - 20,
            10,
            0x22,
            0x2C,
            0x36,
            240,
        );
        let fill_w = ((pw - 20) as f32 * hud.wizard_progress.clamp(0.0, 1.0)) as i32;
        fill_rect(
            buf,
            w,
            h,
            x + 10,
            y + 130,
            fill_w,
            10,
            0x4A,
            0xC0,
            0xD8,
            240,
        );
        if !hud.face_chip.is_empty() {
            let chip: String = hud.face_chip.chars().take(18).collect();
            text(buf, w, h, x + 10, y + 150, &chip, 2, 0xC8, 0xD0, 0xDC, 240);
        }
        if hud.wizard_can_next {
            paint_btn(
                buf,
                w,
                h,
                wizard_next_hit(px, py),
                off_x,
                off_y,
                if hud.wizard_can_add { "OK" } else { "NEXT" },
                true,
            );
        }
        if hud.wizard_can_skip {
            paint_btn(
                buf,
                w,
                h,
                wizard_skip_hit(px, py),
                off_x,
                off_y,
                "SKIP",
                false,
            );
        }
        if hud.wizard_can_add {
            paint_btn(
                buf,
                w,
                h,
                wizard_add_hit(px, py),
                off_x,
                off_y,
                "ADD",
                false,
            );
        }
        let _ = (pw, ph);
        return;
    }

    paint_btn(
        buf,
        w,
        h,
        tab_look_hit(px, py),
        off_x,
        off_y,
        "UX",
        settings.page == SettingsPage::Look && !hud.enroll_active,
    );
    paint_btn(
        buf,
        w,
        h,
        tab_people_hit(px, py),
        off_x,
        off_y,
        "ID",
        settings.page == SettingsPage::People || hud.enroll_active,
    );

    if hud.enroll_active {
        text(
            buf,
            w,
            h,
            x + 10,
            y + 64,
            "ENROLL",
            2,
            0xB0,
            0xC0,
            0xD0,
            240,
        );
        let hint: String = hud.enroll_hint.chars().take(16).collect();
        text(buf, w, h, x + 10, y + 92, &hint, 2, 0x7E, 0xE8, 0xFF, 250);
        fill_rect(
            buf,
            w,
            h,
            x + 10,
            y + 130,
            pw - 20,
            10,
            0x22,
            0x2C,
            0x36,
            240,
        );
        let fill_w = ((pw - 20) as f32 * hud.enroll_progress.clamp(0.0, 1.0)) as i32;
        fill_rect(
            buf,
            w,
            h,
            x + 10,
            y + 130,
            fill_w,
            10,
            0x4A,
            0xC0,
            0xD8,
            240,
        );
        if !hud.face_chip.is_empty() {
            let chip: String = hud.face_chip.chars().take(18).collect();
            text(buf, w, h, x + 10, y + 148, &chip, 2, 0xC8, 0xD0, 0xDC, 240);
        }
        paint_btn(
            buf,
            w,
            h,
            cancel_enroll_hit(px, py),
            off_x,
            off_y,
            "CANCEL",
            false,
        );
        let _ = (pw, ph);
        return;
    }

    if settings.page == SettingsPage::People {
        text(
            buf,
            w,
            h,
            x + 10,
            y + 64,
            "PEOPLE",
            2,
            0xB0,
            0xC0,
            0xD0,
            240,
        );
        if hud.people.is_empty() {
            text(
                buf,
                w,
                h,
                x + 10,
                y + 92,
                "NONE YET",
                2,
                0x8A,
                0x9A,
                0xA8,
                230,
            );
        }
        for (i, name) in hud.people.iter().take(4).enumerate() {
            let r = person_row_hit(px, py, i);
            let on = i == settings.selected_person;
            paint_btn(
                buf,
                w,
                h,
                r,
                off_x,
                off_y,
                &name.chars().take(10).collect::<String>(),
                on,
            );
        }
        paint_btn(buf, w, h, id_btn(px, py, 0), off_x, off_y, "FACE", false);
        paint_btn(buf, w, h, id_btn(px, py, 1), off_x, off_y, "VOICE", false);
        paint_btn(buf, w, h, id_btn(px, py, 2), off_x, off_y, "HAND", false);
        paint_btn(buf, w, h, new_btn(px, py), off_x, off_y, "ADD", false);
        paint_btn(buf, w, h, del_btn(px, py), off_x, off_y, "DEL", false);
        paint_btn(
            buf,
            w,
            h,
            need_face_hit(px, py),
            off_x,
            off_y,
            if settings.gestures_need_face {
                "HANDS NEED FACE"
            } else {
                "HANDS OPEN"
            },
            settings.gestures_need_face,
        );
        let _ = (pw, ph);
        return;
    }

    text(buf, w, h, x + 10, y + 64, "CAM", 2, 0xB0, 0xC0, 0xD0, 240);
    let cam = camera_hit(px, py);
    paint_btn(
        buf,
        w,
        h,
        cam,
        off_x,
        off_y,
        if settings.camera { "ON" } else { "OFF" },
        settings.camera,
    );

    text(
        buf,
        w,
        h,
        x + 10,
        y + 64 + 28,
        "GATE",
        2,
        0xB0,
        0xC0,
        0xD0,
        240,
    );
    // gate button sits on the CAM row's right in hit tests; label it beside CAM
    let _ = gate_hit(px, py);
    // Place GATE label on look page under CAM via reused camera row + extra.
    text(
        buf,
        w,
        h,
        x + 168,
        y + 64,
        settings.gate.label(),
        2,
        0x7E,
        0xE8,
        0xFF,
        250,
    );

    fn row(buf: &mut [u8], w: u16, h: u16, x: i32, y: i32, label: &str, t: f32, value: &str) {
        text(buf, w, h, x + 10, y, label, 2, 0xB0, 0xC0, 0xD0, 240);
        fill_rect(
            buf,
            w,
            h,
            x + 88,
            y + 2,
            (PANEL_W as i32) - 104,
            8,
            0x22,
            0x2C,
            0x36,
            240,
        );
        let knob = 88 + (((PANEL_W as i32) - 104) as f32 * t).round() as i32;
        fill_rect(
            buf,
            w,
            h,
            x + 88,
            y + 2,
            knob - 88,
            8,
            0x4A,
            0xC0,
            0xD8,
            230,
        );
        fill_disk(
            buf,
            w,
            h,
            (x + knob) as f32,
            (y + 6) as f32,
            5.0,
            0x7E,
            0xE8,
            0xFF,
            250,
        );
        text(buf, w, h, x + 88, y - 10, value, 1, 0x8A, 0x9A, 0xA8, 220);
    }

    let lt = (settings.listen_ms as f32 - 2_000.0) / 13_000.0;
    let listen = slider_track(px, py, 0);
    row(
        buf,
        w,
        h,
        x,
        (listen.1 - off_y) as i32,
        "LISTEN",
        lt,
        &format!("{:.1}S", settings.listen_ms as f32 / 1000.0),
    );
    let gt = (settings.gaze_lock_ms as f32 - 2_000.0) / 13_000.0;
    let gaze = slider_track(px, py, 1);
    row(
        buf,
        w,
        h,
        x,
        (gaze.1 - off_y) as i32,
        "GAZE",
        gt,
        &format!("{:.1}S", settings.gaze_lock_ms as f32 / 1000.0),
    );
    let st = (settings.stroke - 0.8) / 2.2;
    let stroke_tr = slider_track(px, py, 2);
    row(
        buf,
        w,
        h,
        x,
        (stroke_tr.1 - off_y) as i32,
        "STROKE",
        st,
        &format!("{:.1}", settings.stroke),
    );
    let calib_on = hud.gaze_calib_open || hud.gaze_calib_hint == "RECALIBRATE";
    paint_btn(
        buf,
        w,
        h,
        calib_hit(px, py),
        off_x,
        off_y,
        "CALIB",
        calib_on,
    );
    let _ = (pw, ph);
}

pub fn paint_auth_chip(
    buf: &mut [u8],
    w: u16,
    h: u16,
    cx: f32,
    cy: f32,
    label: &str,
    known: bool,
    off_x: f32,
    off_y: f32,
) {
    let text_w = (label.len() as i32 * 8).max(24);
    let x = (cx - off_x).round() as i32 - text_w / 2;
    let y = (cy - off_y - 118.0).round() as i32;
    fill_rect(
        buf,
        w,
        h,
        x - 4,
        y - 3,
        text_w + 8,
        14,
        if known { 0x14 } else { 0x28 },
        if known { 0x3A } else { 0x22 },
        if known { 0x48 } else { 0x28 },
        230,
    );
    text(
        buf,
        w,
        h,
        x,
        y,
        label,
        2,
        if known { 0x7E } else { 0xD0 },
        if known { 0xE8 } else { 0xA0 },
        if known { 0xFF } else { 0x90 },
        250,
    );
}

pub fn paint_listening_chrome(
    buf: &mut [u8],
    w: u16,
    h: u16,
    cx: f32,
    cy: f32,
    menu: &RadialMenu,
    settings: &Settings,
    hud: &IdentityHud,
    mx: f32,
    my: f32,
    now_ms: u64,
    screen_w: f32,
    screen_h: f32,
    off_x: f32,
    off_y: f32,
) {
    let centers = icon_centers(cx, cy, menu.layout_ms(now_ms), true);
    let hot = crate::radial::hit_icon(mx, my, &centers);
    for &(id, x, y) in &centers {
        paint_icon(
            buf,
            w,
            h,
            x,
            y,
            id,
            hot == Some(id) || (id == IconId::Settings && menu.settings_open),
            menu.flashing(id, now_ms),
            off_x,
            off_y,
        );
    }
    if menu.settings_open || menu.wizard_open {
        if let Some((_, sx, sy)) = centers
            .iter()
            .copied()
            .find(|(id, _, _)| *id == IconId::Settings)
        {
            let (px, py, _, _) = panel_rect(sx, sy, screen_w, screen_h);
            paint_panel(buf, w, h, px, py, settings, hud, off_x, off_y);
        }
    }
    if menu.calib_open || hud.gaze_calib_open {
        paint_gaze_calib(buf, w, h, cx, cy, hud, screen_w, screen_h, off_x, off_y);
    }
}

fn paint_gaze_calib(
    buf: &mut [u8],
    w: u16,
    h: u16,
    cx: f32,
    cy: f32,
    hud: &IdentityHud,
    screen_w: f32,
    screen_h: f32,
    off_x: f32,
    off_y: f32,
) {
    let n = crate::identity::calib_point_count();
    let n = if n == 9 { 9 } else { 5 };
    for i in 0..n {
        let (dx, dy) = crate::identity::dot_pos(i, n, screen_w, screen_h);
        let hot = hud.gaze_calib_hint.starts_with("HOLD")
            || crate::identity::look_hint(i, n) == hud.gaze_calib_hint;
        let rad = if hot { 14.0 } else { 9.0 };
        fill_disk(
            buf,
            w,
            h,
            dx - off_x,
            dy - off_y,
            rad + 2.0,
            0x0C,
            0x14,
            0x1C,
            220,
        );
        fill_disk(
            buf,
            w,
            h,
            dx - off_x,
            dy - off_y,
            rad,
            if hot { 0x7E } else { 0x3A },
            if hot { 0xE8 } else { 0x48 },
            if hot { 0xFF } else { 0x58 },
            240,
        );
        if hot && hud.gaze_calib_progress > 0.02 {
            let p = hud.gaze_calib_progress.clamp(0.0, 1.0);
            fill_disk(
                buf,
                w,
                h,
                dx - off_x,
                dy - off_y,
                rad * p,
                0xF4,
                0xFB,
                0xFF,
                200,
            );
        }
    }
    let hint: String = hud.gaze_calib_hint.chars().take(16).collect();
    if !hint.is_empty() {
        let tw = (hint.len() as i32 * 8).max(24);
        let x = (cx - off_x).round() as i32 - tw / 2;
        let y = (cy - off_y + 86.0).round() as i32;
        fill_rect(buf, w, h, x - 4, y - 3, tw + 8, 14, 0x12, 0x18, 0x22, 230);
        text(buf, w, h, x, y, &hint, 2, 0x7E, 0xE8, 0xFF, 250);
    }
    let ((sx, sy), (cx_, cy_)) = calib_chip_centers(cx, cy);
    fill_disk(
        buf,
        w,
        h,
        sx - off_x,
        sy - off_y,
        ICON_RADIUS,
        0x2A,
        0x3A,
        0x48,
        240,
    );
    fill_disk(
        buf,
        w,
        h,
        cx_ - off_x,
        cy_ - off_y,
        ICON_RADIUS,
        0x2A,
        0x3A,
        0x48,
        240,
    );
    text(
        buf,
        w,
        h,
        (sx - off_x - 14.0).round() as i32,
        (sy - off_y - 4.0).round() as i32,
        "SKIP",
        2,
        0xE8,
        0xF4,
        0xFC,
        250,
    );
    text(
        buf,
        w,
        h,
        (cx_ - off_x - 22.0).round() as i32,
        (cy_ - off_y - 4.0).round() as i32,
        "CANCEL",
        2,
        0xE8,
        0xF4,
        0xFC,
        250,
    );
}

/// Paint the buddy (and listening chrome) into a software ARGB buffer.
/// `off_x`/`off_y` are the buffer origin in screen space (dirty-rect crop).
pub fn paint_overlay(
    buf: &mut [u8],
    w: u16,
    h: u16,
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
    off_x: f32,
    off_y: f32,
) {
    use crate::sim::HIT_RADIUS;
    paint_wireframe(buf, w, h, state, off_x, off_y, listening, settings.stroke);
    if !listening && dwell > 0.04 {
        let rad = HIT_RADIUS * (0.62 + 0.28 * dwell);
        let alpha = (36.0 + 90.0 * dwell) as u8;
        const N: i32 = 64;
        for i in 0..N {
            let t0 = i as f32 * (std::f32::consts::TAU / N as f32);
            let t1 = (i + 1) as f32 * (std::f32::consts::TAU / N as f32);
            stroke(
                buf,
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
            stroke(
                buf,
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
        paint_listening_chrome(
            buf, w, h, state.cx, state.cy, menu, settings, hud, mx, my, now_ms, screen_w, screen_h,
            off_x, off_y,
        );
    }
    let known = hud.auth_label != "UNKNOWN" && !hud.auth_label.is_empty();
    if !hud.auth_label.is_empty() {
        paint_auth_chip(
            buf,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::initial;

    #[test]
    fn depth_sort_is_far_to_near() {
        let s = initial();
        let e = edges_far_to_near(&s);
        for i in 1..e.len() {
            assert!(
                e[i].4 + 1e-5 >= e[i - 1].4,
                "{} then {}",
                e[i - 1].4,
                e[i].4
            );
        }
    }

    #[test]
    fn later_stroke_covers_earlier() {
        let mut buf = vec![0u8; 32 * 32 * 4];
        stroke(
            &mut buf, 32, 32, 4.0, 16.0, 28.0, 16.0, 3.0, 10, 10, 10, 255,
        );
        stroke(
            &mut buf, 32, 32, 16.0, 4.0, 16.0, 28.0, 3.0, 200, 200, 200, 255,
        );
        let i = (16 * 32 + 16) * 4;
        // Near-white core should win at the crossing.
        assert!(buf[i + 2] > 120, "r={}", buf[i + 2]);
    }

    #[test]
    fn paint_overlay_writes_buddy_pixels() {
        let s = crate::sim::initial_on(400.0, 300.0);
        let menu = crate::radial::RadialMenu::new();
        let settings = crate::radial::Settings::default();
        let hud = crate::radial::IdentityHud {
            auth_label: "ADA".into(),
            ..Default::default()
        };
        let mut buf = vec![0u8; 200 * 200 * 4];
        paint_overlay(
            &mut buf,
            200,
            200,
            &s,
            false,
            0.0,
            0.0,
            &menu,
            &settings,
            &hud,
            0.0,
            0.0,
            0,
            400.0,
            300.0,
            s.cx - 100.0,
            s.cy - 100.0,
        );
        assert!(buf.iter().any(|b| *b != 0), "overlay should paint strokes");
    }
}
