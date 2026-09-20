//! MediaPipe-style palm SSD anchors + decode (always compiled).
//!
//! Targets the OpenCV zoo / PINTO 192×192 palm detector (2016 anchors).
//! Landmark crop uses the wrist→middle-finger axis, then a 224×224 ROI
//! for the hand-landmark ONNX.

pub const PALM_INPUT: u32 = 192;
pub const HAND_LANDMARK_SIZE: u32 = 224;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PalmDet {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
    pub kps: [(f32, f32); 7],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HandRoi {
    pub cx: f32,
    pub cy: f32,
    pub size: f32,
    pub angle: f32,
}

fn scale_at(min_s: f32, max_s: f32, i: usize, n: usize) -> f32 {
    if n <= 1 {
        return (min_s + max_s) * 0.5;
    }
    min_s + (max_s - min_s) * i as f32 / (n as f32 - 1.0)
}

/// MediaPipe SSD anchors. `aspects = [1.0, 0.5]`, no interpolated extra,
/// 192 input, strides `[8,16,16,16]` → 2016 boxes (OpenCV zoo 2023feb).
pub fn generate_palm_anchors(input: u32, strides: &[u32], aspects: &[f32]) -> Vec<Anchor> {
    let mut anchors = Vec::new();
    let n = strides.len();
    for (layer, &stride) in strides.iter().enumerate() {
        if stride == 0 {
            continue;
        }
        let fm = (input / stride).max(1);
        let _scale = scale_at(0.1484375, 0.75, layer, n);
        let _ = _scale;
        for y in 0..fm {
            for x in 0..fm {
                for _ar in aspects {
                    let x_center = (x as f32 + 0.5) / fm as f32;
                    let y_center = (y as f32 + 0.5) / fm as f32;
                    anchors.push(Anchor {
                        x: x_center,
                        y: y_center,
                        w: 1.0,
                        h: 1.0,
                    });
                }
            }
        }
    }
    anchors
}

pub fn palm_anchors_192() -> Vec<Anchor> {
    generate_palm_anchors(192, &[8, 16, 16, 16], &[1.0, 0.5])
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Decode one palm. `reg` is 18 floats (dx,dy,w,h + 7 keypoints) in
/// **input-pixel** offset form used by MediaPipe / OpenCV zoo.
pub fn decode_palm_box(
    reg: &[f32],
    score_logit: f32,
    anchor: Anchor,
    input: f32,
) -> Option<PalmDet> {
    if reg.len() < 18 {
        return None;
    }
    let score = sigmoid(score_logit);
    let ax = anchor.x * input;
    let ay = anchor.y * input;
    let cx = reg[0] + ax;
    let cy = reg[1] + ay;
    let w = reg[2];
    let h = reg[3];
    let mut kps = [(0.0f32, 0.0f32); 7];
    for i in 0..7 {
        kps[i] = (reg[4 + i * 2] + ax, reg[5 + i * 2] + ay);
    }
    Some(PalmDet {
        x: cx - w * 0.5,
        y: cy - h * 0.5,
        w: w.max(1.0),
        h: h.max(1.0),
        score,
        kps,
    })
}

pub fn decode_palms(
    regressors: &[f32],
    scores: &[f32],
    anchors: &[Anchor],
    input: f32,
    thresh: f32,
) -> Vec<PalmDet> {
    let n = scores.len().min(anchors.len()).min(regressors.len() / 18);
    let mut out = Vec::new();
    for i in 0..n {
        let det = match decode_palm_box(
            &regressors[i * 18..i * 18 + 18],
            scores[i],
            anchors[i],
            input,
        ) {
            Some(d) => d,
            None => continue,
        };
        if det.score >= thresh {
            out.push(det);
        }
    }
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Wrist (kps 0) → middle MCP (kps 2). Size ≈ 2.6 × max(box side).
pub fn palm_to_roi(palm: &PalmDet, frame_w: f32, frame_h: f32) -> HandRoi {
    let wrist = palm.kps[0];
    let middle = palm.kps[2];
    let angle = -(middle.1 - wrist.1).atan2(middle.0 - wrist.0);
    let size = palm.w.max(palm.h) * 2.6;
    let cx = (palm.x + palm.w * 0.5).clamp(0.0, frame_w);
    let cy = (palm.y + palm.h * 0.5).clamp(0.0, frame_h);
    HandRoi {
        cx,
        cy,
        size: size.max(24.0),
        angle,
    }
}

/// Sample `out×out` RGB from a rotated square ROI (backward warp).
pub fn warp_roi_rgb(rgb: &[u8], w: u32, h: u32, roi: HandRoi, out: usize) -> Vec<u8> {
    let mut dst = vec![0u8; out * out * 3];
    if w == 0 || h == 0 || roi.size <= 1.0 {
        return dst;
    }
    let (sa, ca) = roi.angle.sin_cos();
    for y in 0..out {
        for x in 0..out {
            let nx = (x as f32 + 0.5) / out as f32 - 0.5;
            let ny = (y as f32 + 0.5) / out as f32 - 0.5;
            let rx = nx * roi.size;
            let ry = ny * roi.size;
            let sx = roi.cx + rx * ca - ry * sa;
            let sy = roi.cy + rx * sa + ry * ca;
            if sx < 0.0 || sy < 0.0 || sx >= w as f32 || sy >= h as f32 {
                continue;
            }
            let ix = sx.floor() as u32;
            let iy = sy.floor() as u32;
            let i = ((iy * w + ix) * 3) as usize;
            let o = (y * out + x) * 3;
            if i + 2 < rgb.len() {
                dst[o] = rgb[i];
                dst[o + 1] = rgb[i + 1];
                dst[o + 2] = rgb[i + 2];
            }
        }
    }
    dst
}

/// Map a landmark in `out×out` crop space back to the original frame.
pub fn roi_to_frame(roi: HandRoi, out: f32, x: f32, y: f32) -> (f32, f32) {
    let nx = x / out - 0.5;
    let ny = y / out - 0.5;
    let (sa, ca) = roi.angle.sin_cos();
    let rx = nx * roi.size;
    let ry = ny * roi.size;
    (roi.cx + rx * ca - ry * sa, roi.cy + rx * sa + ry * ca)
}

pub fn nms_palms(mut dets: Vec<PalmDet>, thresh: f32) -> Vec<PalmDet> {
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = Vec::new();
    let mut skip = vec![false; dets.len()];
    for i in 0..dets.len() {
        if skip[i] {
            continue;
        }
        keep.push(dets[i]);
        for j in i + 1..dets.len() {
            if skip[j] {
                continue;
            }
            let a = &dets[i];
            let b = &dets[j];
            let ix1 = a.x.max(b.x);
            let iy1 = a.y.max(b.y);
            let ix2 = (a.x + a.w).min(b.x + b.w);
            let iy2 = (a.y + a.h).min(b.y + b.h);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
            let union = a.w * a.h + b.w * b.h - inter;
            if union > 1e-6 && inter / union > thresh {
                skip[j] = true;
            }
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_2016() {
        let a = palm_anchors_192();
        assert_eq!(a.len(), 2016);
        assert!((a[0].x - 0.5 / 24.0).abs() < 1e-5);
    }

    #[test]
    fn decode_high_score_box() {
        let anchors = palm_anchors_192();
        let mut reg = vec![0.0f32; 18];
        // center offset 0, size 40, kps around center
        reg[2] = 40.0;
        reg[3] = 40.0;
        for i in 0..7 {
            reg[4 + i * 2] = 5.0;
            reg[5 + i * 2] = -3.0;
        }
        let det = decode_palm_box(&reg, 4.0, anchors[0], 192.0).unwrap();
        assert!(det.score > 0.9);
        assert!((det.w - 40.0).abs() < 1e-4);
        assert_eq!(det.kps.len(), 7);
    }

    #[test]
    fn roi_roundtrip_center() {
        let roi = HandRoi {
            cx: 80.0,
            cy: 60.0,
            size: 50.0,
            angle: 0.0,
        };
        let (x, y) = roi_to_frame(roi, 224.0, 112.0, 112.0);
        assert!((x - 80.0).abs() < 1e-3 && (y - 60.0).abs() < 1e-3);
    }

    #[test]
    fn warp_roi_samples_center() {
        let mut rgb = vec![0u8; 32 * 32 * 3];
        rgb[(16 * 32 + 16) * 3] = 180;
        let roi = HandRoi {
            cx: 16.0,
            cy: 16.0,
            size: 8.0,
            angle: 0.0,
        };
        let crop = warp_roi_rgb(&rgb, 32, 32, roi, 8);
        let mid = (4 * 8 + 4) * 3;
        assert!(
            crop[mid] > 100,
            "center of ROI should hit the painted pixel"
        );
    }
}
