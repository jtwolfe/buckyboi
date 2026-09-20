//! SCRFD decode (`det_10g.onnx` / InsightFace buffalo detector).
//!
//! Tensor layout matches `detection/scrfd/tools/scrfd.py`:
//! 9 outputs → score_8/16/32, bbox_8/16/32, kps_8/16/32 (fmc=3, 2 anchors).
//! Boxes are distance-to-boundary from the anchor centre; keypoints are
//! 5×(dx, dy) from the same centre, already in stride units in the raw
//! tensors (we multiply by stride here).
//!
//! ONNX I/O lives behind the `face` feature. Decode / NMS / letterbox /
//! multi-face pick are always compiled so tests do not need models.

pub const SCRFD_SIZE: u32 = 640;
pub const SCRFD_STRIDES: [u32; 3] = [8, 16, 32];
pub const SCRFD_ANCHORS: usize = 2;
pub const SCRFD_DET_THRESH: f32 = 0.5;
pub const SCRFD_NMS_THRESH: f32 = 0.4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectedFace {
    pub bbox: FaceBox,
    pub kps: Option<[(f32, f32); 5]>,
    pub score: f32,
}

impl DetectedFace {
    pub fn cx(&self) -> f32 {
        self.bbox.x + self.bbox.w * 0.5
    }
    pub fn cy(&self) -> f32 {
        self.bbox.y + self.bbox.h * 0.5
    }
    pub fn area(&self) -> f32 {
        self.bbox.w.max(0.0) * self.bbox.h.max(0.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub new_w: u32,
    pub new_h: u32,
    pub scale: f32,
    pub canvas: u32,
}

/// InsightFace letterbox: keep aspect, pad bottom/right (top-left aligned).
pub fn letterbox_params(src_w: u32, src_h: u32, canvas: u32) -> Letterbox {
    if src_w == 0 || src_h == 0 {
        return Letterbox {
            new_w: canvas,
            new_h: canvas,
            scale: 1.0,
            canvas,
        };
    }
    let im_ratio = src_h as f32 / src_w as f32;
    let model_ratio = 1.0;
    let (new_w, new_h) = if im_ratio > model_ratio {
        let new_height = canvas;
        let new_width = ((new_height as f32) / im_ratio).round() as u32;
        (new_width.max(1).min(canvas), new_height)
    } else {
        let new_width = canvas;
        let new_height = ((new_width as f32) * im_ratio).round() as u32;
        (new_width, new_height.max(1).min(canvas))
    };
    Letterbox {
        new_w,
        new_h,
        scale: new_h as f32 / src_h as f32,
        canvas,
    }
}

pub fn letterbox_rgb(rgb: &[u8], w: u32, h: u32, lb: Letterbox) -> Vec<u8> {
    let c = lb.canvas as usize;
    let mut out = vec![0u8; c * c * 3];
    if w == 0 || h == 0 || lb.new_w == 0 || lb.new_h == 0 {
        return out;
    }
    for y in 0..lb.new_h as usize {
        let sy = ((y as f32 + 0.5) * h as f32 / lb.new_h as f32).floor() as u32;
        let sy = sy.min(h - 1);
        for x in 0..lb.new_w as usize {
            let sx = ((x as f32 + 0.5) * w as f32 / lb.new_w as f32).floor() as u32;
            let sx = sx.min(w - 1);
            let i = ((sy * w + sx) * 3) as usize;
            let o = (y * c + x) * 3;
            if i + 2 < rgb.len() && o + 2 < out.len() {
                out[o] = rgb[i];
                out[o + 1] = rgb[i + 1];
                out[o + 2] = rgb[i + 2];
            }
        }
    }
    out
}

/// One FPN level. `bbox` is `[n, 4]` (l,t,r,b) in stride units; `kps` is `[n, 10]`.
pub fn decode_level(
    scores: &[f32],
    bbox: &[f32],
    kps: Option<&[f32]>,
    stride: u32,
    feat_h: usize,
    feat_w: usize,
    num_anchors: usize,
    thresh: f32,
    scale: f32,
) -> Vec<DetectedFace> {
    let mut out = Vec::new();
    if feat_h == 0 || feat_w == 0 || num_anchors == 0 || stride == 0 {
        return out;
    }
    let cells = feat_h * feat_w;
    let n = cells * num_anchors;
    let stride_f = stride as f32;
    for anchor_idx in 0..n {
        let score = *scores.get(anchor_idx).unwrap_or(&0.0);
        if score < thresh {
            continue;
        }
        let cell = anchor_idx / num_anchors;
        let gy = cell / feat_w;
        let gx = cell % feat_w;
        let cx = gx as f32 * stride_f;
        let cy = gy as f32 * stride_f;
        let b = anchor_idx * 4;
        if b + 3 >= bbox.len() {
            continue;
        }
        let x1 = (cx - bbox[b] * stride_f) / scale;
        let y1 = (cy - bbox[b + 1] * stride_f) / scale;
        let x2 = (cx + bbox[b + 2] * stride_f) / scale;
        let y2 = (cy + bbox[b + 3] * stride_f) / scale;
        let mut kps_out = None;
        if let Some(k) = kps {
            let k0 = anchor_idx * 10;
            if k0 + 9 < k.len() {
                let mut pts = [(0.0f32, 0.0f32); 5];
                for p in 0..5 {
                    pts[p] = (
                        (cx + k[k0 + p * 2] * stride_f) / scale,
                        (cy + k[k0 + p * 2 + 1] * stride_f) / scale,
                    );
                }
                kps_out = Some(pts);
            }
        }
        out.push(DetectedFace {
            bbox: FaceBox {
                x: x1,
                y: y1,
                w: (x2 - x1).max(1.0),
                h: (y2 - y1).max(1.0),
            },
            kps: kps_out,
            score,
        });
    }
    out
}

fn iou(a: &FaceBox, b: &FaceBox) -> f32 {
    let ax2 = a.x + a.w;
    let ay2 = a.y + a.h;
    let bx2 = b.x + b.w;
    let by2 = b.y + b.h;
    let ix1 = a.x.max(b.x);
    let iy1 = a.y.max(b.y);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 1e-6 {
        0.0
    } else {
        inter / union
    }
}

pub fn nms(mut faces: Vec<DetectedFace>, thresh: f32) -> Vec<DetectedFace> {
    faces.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = Vec::new();
    let mut suppressed = vec![false; faces.len()];
    for i in 0..faces.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(faces[i]);
        for j in i + 1..faces.len() {
            if !suppressed[j] && iou(&faces[i].bbox, &faces[j].bbox) > thresh {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// InsightFace default: `area − 2 · offset_dist²`. Largest / most central wins.
pub fn pick_primary_face(
    faces: &[DetectedFace],
    frame_w: f32,
    frame_h: f32,
) -> Option<DetectedFace> {
    if faces.is_empty() {
        return None;
    }
    let cx = frame_w * 0.5;
    let cy = frame_h * 0.5;
    faces
        .iter()
        .max_by(|a, b| {
            let va = a.area() - 2.0 * (a.cx() - cx).hypot(a.cy() - cy).powi(2);
            let vb = b.area() - 2.0 * (b.cx() - cx).hypot(b.cy() - cy).powi(2);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
}

/// Flattened ONNX tensors: `(shape, data)` in model output order.
#[derive(Clone, Debug)]
pub struct ScrfdTensor {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

/// Parse 6 or 9 (or 10/15) SCRFD outputs into detections in **letterboxed** space,
/// then divide by `scale` so boxes land in the original frame.
pub fn parse_scrfd_outputs(
    tensors: &[ScrfdTensor],
    input_h: u32,
    input_w: u32,
    scale: f32,
    thresh: f32,
) -> Vec<DetectedFace> {
    let n = tensors.len();
    const S3: &[u32] = &[8, 16, 32];
    const S5: &[u32] = &[8, 16, 32, 64, 128];
    let (fmc, strides, num_anchors, use_kps): (usize, &[u32], usize, bool) = match n {
        6 => (3, S3, 2, false),
        9 => (3, S3, 2, true),
        10 => (5, S5, 1, false),
        15 => (5, S5, 1, true),
        _ => return Vec::new(),
    };
    let mut faces = Vec::new();
    for (idx, stride) in strides.iter().copied().enumerate() {
        if idx >= fmc || idx >= tensors.len() {
            break;
        }
        let feat_h = (input_h / stride) as usize;
        let feat_w = (input_w / stride) as usize;
        let scores = &tensors[idx].data;
        let bbox_t = tensors.get(idx + fmc);
        let Some(bbox_t) = bbox_t else {
            continue;
        };
        let kps_t = if use_kps {
            tensors.get(idx + fmc * 2).map(|t| t.data.as_slice())
        } else {
            None
        };
        faces.extend(decode_level(
            scores,
            &bbox_t.data,
            kps_t,
            stride,
            feat_h,
            feat_w,
            num_anchors,
            thresh,
            scale,
        ));
        let _ = (feat_h, feat_w);
    }
    nms(faces, SCRFD_NMS_THRESH)
}

pub fn det_model_names() -> &'static [&'static str] {
    &[
        "det_10g.onnx",
        "scrfd_10g.onnx",
        "buffalo_l_det_10g.onnx",
        "det_500m.onnx",
        "scrfd.onnx",
    ]
}

#[cfg(feature = "face")]
pub fn find_det_model() -> Option<std::path::PathBuf> {
    let dir = crate::identity::models_dir()?;
    for name in det_model_names() {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[cfg(feature = "face")]
mod onnx {
    use super::*;
    use ndarray::Array4;
    use std::sync::Mutex;

    struct Det {
        session: ort::session::Session,
    }

    static DET: Mutex<Option<Det>> = Mutex::new(None);

    fn session() -> Option<()> {
        let mut g = DET.lock().ok()?;
        if g.is_some() {
            return Some(());
        }
        let path = find_det_model()?;
        let sess = ort::session::Session::builder()
            .ok()?
            .commit_from_file(&path)
            .ok()?;
        *g = Some(Det { session: sess });
        eprintln!("buckyboi: SCRFD detector {}", path.display());
        Some(())
    }

    pub fn detect(rgb: &[u8], w: u32, h: u32) -> Option<Vec<DetectedFace>> {
        session()?;
        let mut g = DET.lock().ok()?;
        let det = g.as_mut()?;
        let lb = letterbox_params(w, h, SCRFD_SIZE);
        let canvas = letterbox_rgb(rgb, w, h, lb);
        // RGB → BGR blob, (x-127.5)/128. InsightFace `blobFromImage(..., swapRB=True)`
        // on a BGR `imread` is RGB in the blob; our frames are already RGB, so we
        // still emit BGR-first NCHW to match the published buffalo_l graph
        // (channel 0 = B). Official Python uses swapRB=True on BGR = RGB-first.
        // We follow the Python blob: RGB-first after swapRB.
        let size = SCRFD_SIZE as usize;
        let mut blob = Array4::<f32>::zeros((1, 3, size, size));
        for y in 0..size {
            for x in 0..size {
                let i = (y * size + x) * 3;
                if i + 2 >= canvas.len() {
                    continue;
                }
                // RGB-first (swapRB on BGR).
                blob[[0, 0, y, x]] = (canvas[i] as f32 - 127.5) / 128.0;
                blob[[0, 1, y, x]] = (canvas[i + 1] as f32 - 127.5) / 128.0;
                blob[[0, 2, y, x]] = (canvas[i + 2] as f32 - 127.5) / 128.0;
            }
        }
        let input = ort::value::Tensor::from_array(blob).ok()?;
        let outputs = det.session.run(ort::inputs![input]).ok()?;
        let mut tensors = Vec::new();
        for (_, v) in outputs.iter() {
            if let Ok((shape, data)) = v.try_extract_tensor::<f32>() {
                tensors.push(ScrfdTensor {
                    shape: shape.iter().map(|d| *d as i64).collect(),
                    data: data.to_vec(),
                });
            }
        }
        if tensors.is_empty() {
            return None;
        }
        Some(parse_scrfd_outputs(
            &tensors,
            SCRFD_SIZE,
            SCRFD_SIZE,
            lb.scale,
            SCRFD_DET_THRESH,
        ))
    }
}

#[cfg(feature = "face")]
pub fn detect_onnx(rgb: &[u8], w: u32, h: u32) -> Option<Vec<DetectedFace>> {
    onnx::detect(rgb, w, h)
}

#[cfg(not(feature = "face"))]
pub fn detect_onnx(_rgb: &[u8], _w: u32, _h: u32) -> Option<Vec<DetectedFace>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_320x240() {
        let lb = letterbox_params(320, 240, 640);
        assert_eq!(lb.new_w, 640);
        assert_eq!(lb.new_h, 480);
        assert!((lb.scale - 2.0).abs() < 1e-5);
    }

    #[test]
    fn decode_known_anchor() {
        // 16×16 input, stride 8 → 2×2 cells, 2 anchors = 8 preds.
        // Cell (1,1) → center (8, 8). Distances 2 (stride units) → 16 px box.
        let mut scores = vec![0.0f32; 8];
        let mut bbox = vec![0.0f32; 32];
        let mut kps = vec![0.0f32; 80];
        let idx = (1 * 2 + 1) * 2; // cell (y=1,x=1), anchor 0
        scores[idx] = 0.91;
        // 1 stride-unit of distance * stride 8 = 8 px → box (0,0)-(16,16)
        bbox[idx * 4] = 1.0;
        bbox[idx * 4 + 1] = 1.0;
        bbox[idx * 4 + 2] = 1.0;
        bbox[idx * 4 + 3] = 1.0;
        // left-eye offset (−1, −1) stride units → (0, 0) in letterbox
        kps[idx * 10] = -1.0;
        kps[idx * 10 + 1] = -1.0;
        let faces = decode_level(&scores, &bbox, Some(&kps), 8, 2, 2, 2, 0.5, 1.0);
        assert_eq!(faces.len(), 1);
        let f = faces[0];
        assert!((f.bbox.x - 0.0).abs() < 1e-3, "{:?}", f.bbox);
        assert!((f.bbox.y - 0.0).abs() < 1e-3, "{:?}", f.bbox);
        assert!((f.bbox.w - 16.0).abs() < 1e-3, "{:?}", f.bbox);
        assert!((f.bbox.h - 16.0).abs() < 1e-3, "{:?}", f.bbox);
        let k = f.kps.expect("kps");
        assert!((k[0].0 - 0.0).abs() < 1e-3 && (k[0].1 - 0.0).abs() < 1e-3);
    }

    #[test]
    fn nms_drops_overlap() {
        let a = DetectedFace {
            bbox: FaceBox {
                x: 0.0,
                y: 0.0,
                w: 10.0,
                h: 10.0,
            },
            kps: None,
            score: 0.9,
        };
        let b = DetectedFace {
            bbox: FaceBox {
                x: 1.0,
                y: 1.0,
                w: 10.0,
                h: 10.0,
            },
            kps: None,
            score: 0.6,
        };
        let c = DetectedFace {
            bbox: FaceBox {
                x: 50.0,
                y: 50.0,
                w: 8.0,
                h: 8.0,
            },
            kps: None,
            score: 0.7,
        };
        let kept = nms(vec![a, b, c], 0.4);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 1e-6);
        assert!((kept[1].score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn pick_prefers_large_central() {
        let small_center = DetectedFace {
            bbox: FaceBox {
                x: 45.0,
                y: 45.0,
                w: 10.0,
                h: 10.0,
            },
            kps: None,
            score: 0.99,
        };
        let large_edge = DetectedFace {
            bbox: FaceBox {
                x: 0.0,
                y: 0.0,
                w: 30.0,
                h: 30.0,
            },
            kps: None,
            score: 0.8,
        };
        let large_center = DetectedFace {
            bbox: FaceBox {
                x: 35.0,
                y: 35.0,
                w: 30.0,
                h: 30.0,
            },
            kps: None,
            score: 0.7,
        };
        let pick =
            pick_primary_face(&[small_center, large_edge, large_center], 100.0, 100.0).unwrap();
        assert!((pick.bbox.x - 35.0).abs() < 1e-3);
    }

    #[test]
    fn parse_nine_tensors_mocked() {
        // Tiny 16×16 / stride-8-only signal stuffed into a 9-tensor pack.
        // Other levels stay empty (scores below thresh).
        let mut tensors = Vec::new();
        for _ in 0..9 {
            tensors.push(ScrfdTensor {
                shape: vec![1, 8, 1],
                data: vec![0.0; 8],
            });
        }
        tensors[0].data = vec![0.0; 8];
        tensors[0].data[0] = 0.8;
        tensors[3] = ScrfdTensor {
            shape: vec![1, 8, 4],
            data: vec![1.0; 32],
        };
        tensors[6] = ScrfdTensor {
            shape: vec![1, 8, 10],
            data: vec![0.0; 80],
        };
        let faces = parse_scrfd_outputs(&tensors, 16, 16, 1.0, 0.5);
        assert!(!faces.is_empty());
        assert!(faces[0].kps.is_some());
    }
}
