//! InsightFace ArcFace 5-point similarity alignment (`norm_crop` / `arcface_dst`).
//!
//! Matches `insightface/utils/face_align.py`: estimate a 2-D similarity
//! (scale + rotation + translation) from detected landmarks to the 112×112
//! template, then `warpAffine` with border 0. This is the preprocess ArcFace
//! was trained on — a bbox resize is the usual accuracy killer.

/// InsightFace `arcface_dst` for 112×112 (`norm_crop`, mode=`arcface`).
/// Order: left eye, right eye, nose, left mouth, right mouth.
pub const ARCFACE_DST: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

pub const ARCFACE_SIZE: usize = 112;

/// 2-D similarity: `[x'] = [a, -b, tx] [x]`
///                 `[y']   [b,  a, ty] [y]`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Similarity {
    pub a: f32,
    pub b: f32,
    pub tx: f32,
    pub ty: f32,
}

impl Similarity {
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        (self.a * x - self.b * y + self.tx, self.b * x + self.a * y + self.ty)
    }

    /// Inverse map (dest → src) for backward warping.
    pub fn invert_apply(&self, xp: f32, yp: f32) -> (f32, f32) {
        let det = self.a * self.a + self.b * self.b;
        if det < 1e-12 {
            return (xp, yp);
        }
        let dx = xp - self.tx;
        let dy = yp - self.ty;
        ((self.a * dx + self.b * dy) / det, (-self.b * dx + self.a * dy) / det)
    }
}

/// Least-squares 2-D similarity (Umeyama, no reflection). `src` → `dst`.
pub fn estimate_similarity(src: &[(f32, f32)], dst: &[(f32, f32)]) -> Option<Similarity> {
    let n = src.len().min(dst.len());
    if n < 2 {
        return None;
    }
    let nf = n as f32;
    let mut msx = 0.0;
    let mut msy = 0.0;
    let mut mdx = 0.0;
    let mut mdy = 0.0;
    for i in 0..n {
        msx += src[i].0;
        msy += src[i].1;
        mdx += dst[i].0;
        mdy += dst[i].1;
    }
    msx /= nf;
    msy /= nf;
    mdx /= nf;
    mdy /= nf;
    let mut dot = 0.0;
    let mut cross = 0.0;
    let mut norm_src = 0.0;
    for i in 0..n {
        let sx = src[i].0 - msx;
        let sy = src[i].1 - msy;
        let dx = dst[i].0 - mdx;
        let dy = dst[i].1 - mdy;
        dot += sx * dx + sy * dy;
        cross += sx * dy - sy * dx;
        norm_src += sx * sx + sy * sy;
    }
    if norm_src < 1e-12 {
        return None;
    }
    let a = dot / norm_src;
    let b = cross / norm_src;
    Some(Similarity {
        a,
        b,
        tx: mdx - a * msx + b * msy,
        ty: mdy - b * msx - a * msy,
    })
}

/// Template for `image_size` 112 (InsightFace `estimate_norm`).
pub fn arcface_template(image_size: usize) -> [(f32, f32); 5] {
    if image_size == 0 || image_size % 112 == 0 {
        let ratio = (image_size.max(1) as f32) / 112.0;
        let mut dst = ARCFACE_DST;
        for p in &mut dst {
            p.0 *= ratio;
            p.1 *= ratio;
        }
        dst
    } else {
        // 128-family: ratio vs 128 and +8 px x-shift (InsightFace).
        let ratio = image_size as f32 / 128.0;
        let diff_x = 8.0 * ratio;
        let mut dst = ARCFACE_DST;
        for p in &mut dst {
            p.0 = p.0 * ratio + diff_x;
            p.1 *= ratio;
        }
        dst
    }
}

pub fn estimate_arcface_norm(landmarks: &[(f32, f32); 5], image_size: usize) -> Option<Similarity> {
    let dst = arcface_template(image_size);
    estimate_similarity(landmarks, &dst)
}

fn sample_rgb(rgb: &[u8], w: u32, h: u32, x: f32, y: f32) -> [u8; 3] {
    if w == 0 || h == 0 {
        return [0, 0, 0];
    }
    if !(0.0..w as f32).contains(&x) || !(0.0..h as f32).contains(&y) {
        return [0, 0, 0];
    }
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(w as i32 - 1);
    let y1 = (y0 + 1).min(h as i32 - 1);
    let x0 = x0.clamp(0, w as i32 - 1);
    let y0 = y0.clamp(0, h as i32 - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let pix = |xx: i32, yy: i32| -> [f32; 3] {
        let i = ((yy as u32 * w + xx as u32) * 3) as usize;
        if i + 2 < rgb.len() {
            [rgb[i] as f32, rgb[i + 1] as f32, rgb[i + 2] as f32]
        } else {
            [0.0, 0.0, 0.0]
        }
    };
    let p00 = pix(x0, y0);
    let p10 = pix(x1, y0);
    let p01 = pix(x0, y1);
    let p11 = pix(x1, y1);
    let mut out = [0u8; 3];
    for c in 0..3 {
        let top = p00[c] * (1.0 - tx) + p10[c] * tx;
        let bot = p01[c] * (1.0 - tx) + p11[c] * tx;
        out[c] = (top * (1.0 - ty) + bot * ty).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Backward-warp RGB with border 0 (`cv2.warpAffine(..., borderValue=0)`).
pub fn warp_affine_rgb(
    rgb: &[u8],
    w: u32,
    h: u32,
    m: Similarity,
    out_w: usize,
    out_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; out_w * out_h * 3];
    for y in 0..out_h {
        for x in 0..out_w {
            // OpenCV warpAffine samples dest integer (x, y), not pixel centers.
            let (sx, sy) = m.invert_apply(x as f32, y as f32);
            let p = sample_rgb(rgb, w, h, sx, sy);
            let i = (y * out_w + x) * 3;
            out[i] = p[0];
            out[i + 1] = p[1];
            out[i + 2] = p[2];
        }
    }
    out
}

/// InsightFace `norm_crop`: 112×112 RGB aligned to `arcface_dst`.
pub fn norm_crop_arcface(rgb: &[u8], w: u32, h: u32, landmarks: &[(f32, f32); 5]) -> Option<Vec<u8>> {
    let m = estimate_arcface_norm(landmarks, ARCFACE_SIZE)?;
    Some(warp_affine_rgb(rgb, w, h, m, ARCFACE_SIZE, ARCFACE_SIZE))
}

/// BGR NCHW blob, InsightFace `(x - 127.5) / 128`, from an aligned RGB crop.
pub fn arcface_blob_bgr(aligned_rgb: &[u8], size: usize) -> Vec<f32> {
    let mut blob = vec![0.0f32; 3 * size * size];
    let plane = size * size;
    for y in 0..size {
        for x in 0..size {
            let i = (y * size + x) * 3;
            if i + 2 >= aligned_rgb.len() {
                continue;
            }
            let r = aligned_rgb[i] as f32;
            let g = aligned_rgb[i + 1] as f32;
            let b = aligned_rgb[i + 2] as f32;
            let o = y * size + x;
            blob[o] = (b - 127.5) / 128.0;
            blob[plane + o] = (g - 127.5) / 128.0;
            blob[2 * plane + o] = (r - 127.5) / 128.0;
        }
    }
    blob
}

/// Nearest-neighbor bbox resize into `size`×`size` RGB (degraded ArcFace path).
pub fn resize_box_rgb(
    rgb: &[u8],
    w: u32,
    h: u32,
    x0: f32,
    y0: f32,
    bw: f32,
    bh: f32,
    size: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; size * size * 3];
    if w == 0 || h == 0 || bw <= 1.0 || bh <= 1.0 {
        return out;
    }
    for y in 0..size {
        let fy = y0 + bh * (y as f32 + 0.5) / size as f32;
        let sy = fy.round().clamp(0.0, h as f32 - 1.0) as u32;
        for x in 0..size {
            let fx = x0 + bw * (x as f32 + 0.5) / size as f32;
            let sx = fx.round().clamp(0.0, w as f32 - 1.0) as u32;
            let i = ((sy * w + sx) * 3) as usize;
            let o = (y * size + x) * 3;
            if i + 2 < rgb.len() {
                out[o] = rgb[i];
                out[o + 1] = rgb[i + 1];
                out[o + 2] = rgb[i + 2];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_on_template() {
        let m = estimate_similarity(&ARCFACE_DST, &ARCFACE_DST).unwrap();
        assert!((m.a - 1.0).abs() < 1e-5);
        assert!(m.b.abs() < 1e-5);
        assert!(m.tx.abs() < 1e-4);
        assert!(m.ty.abs() < 1e-4);
    }

    #[test]
    fn scale_and_translate() {
        let src = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)];
        let dst: Vec<(f32, f32)> = src.iter().map(|(x, y)| (x * 2.0 + 3.0, y * 2.0 - 1.0)).collect();
        let m = estimate_similarity(&src, &dst).unwrap();
        assert!((m.a - 2.0).abs() < 1e-5, "{m:?}");
        assert!(m.b.abs() < 1e-5, "{m:?}");
        assert!((m.tx - 3.0).abs() < 1e-4);
        assert!((m.ty + 1.0).abs() < 1e-4);
        let (x, y) = m.apply(1.0, 1.0);
        assert!((x - 5.0).abs() < 1e-4 && (y - 1.0).abs() < 1e-4);
    }

    #[test]
    fn rotate_90_ccw() {
        // (x, y) → (−y, x)
        let src = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 0.0)];
        let dst = [(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 0.0)];
        let m = estimate_similarity(&src, &dst).unwrap();
        assert!(m.a.abs() < 1e-5, "{m:?}");
        assert!((m.b - 1.0).abs() < 1e-5, "{m:?}");
        let (x, y) = m.apply(1.0, 0.0);
        assert!(x.abs() < 1e-4 && (y - 1.0).abs() < 1e-4);
    }

    #[test]
    fn inverse_roundtrip() {
        let m = Similarity {
            a: 1.4,
            b: -0.3,
            tx: 12.0,
            ty: -5.0,
        };
        let (xp, yp) = m.apply(40.0, 70.0);
        let (x, y) = m.invert_apply(xp, yp);
        assert!((x - 40.0).abs() < 1e-4 && (y - 70.0).abs() < 1e-4);
    }

    #[test]
    fn warp_translates_pixel() {
        // 4×4: red at (1, 1). Shift dest so dest (2, 2) samples src (1, 1).
        let mut rgb = vec![0u8; 4 * 4 * 3];
        rgb[(1 * 4 + 1) * 3] = 200;
        let m = Similarity {
            a: 1.0,
            b: 0.0,
            tx: 1.0,
            ty: 1.0,
        };
        let out = warp_affine_rgb(&rgb, 4, 4, m, 4, 4);
        let i = (2 * 4 + 2) * 3;
        assert!(out[i] > 150, "expected translated red, got {}", out[i]);
    }

    #[test]
    fn norm_crop_maps_template_to_itself() {
        // Paint the five dest points on a 112 canvas; aligning those points
        // as if they were detections should keep them near the template.
        let mut rgb = vec![0u8; 112 * 112 * 3];
        for (i, (x, y)) in ARCFACE_DST.iter().enumerate() {
            let xi = x.round() as usize;
            let yi = y.round() as usize;
            let o = (yi * 112 + xi) * 3;
            rgb[o] = 50 + (i as u8) * 40;
            rgb[o + 1] = 200;
            rgb[o + 2] = 10;
        }
        let crop = norm_crop_arcface(&rgb, 112, 112, &ARCFACE_DST).unwrap();
        assert_eq!(crop.len(), 112 * 112 * 3);
        for (x, y) in ARCFACE_DST {
            let o = (y.round() as usize * 112 + x.round() as usize) * 3;
            assert!(crop[o + 1] > 80, "template landmark should stay green");
        }
    }

    #[test]
    fn blob_is_bgr_and_normalized() {
        let mut rgb = vec![0u8; 4];
        rgb.resize(112 * 112 * 3, 0);
        rgb[0] = 255; // R
        rgb[1] = 0;
        rgb[2] = 0;
        let blob = arcface_blob_bgr(&rgb, 112);
        // B plane first: R pixel → B=0 → (0-127.5)/128
        assert!((blob[0] - (0.0 - 127.5) / 128.0).abs() < 1e-5);
        let r_plane = 2 * 112 * 112;
        assert!((blob[r_plane] - (255.0 - 127.5) / 128.0).abs() < 1e-5);
    }
}
