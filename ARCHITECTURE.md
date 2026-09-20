# Architecture

buckyboi is a **Linux-desktop overlay buddy**. Rust is the product. The
Python and Bend trees are historical ports of the original 800×600 window
toy and are not on the overlay / identity path.

```
┌─────────────────────────────────────────────────────────────┐
│  overlay (bin: buckyboi)                                    │
│  Wayland wlr-layer-shell  |  X11 Shape/XFixes               │
│  click-through hit disks, Esc / --quit                      │
│                                                             │
│   display/      backend pick + present + wake socket        │
│   display/wayland.rs   zwlr_layer_shell_v1 overlay + shm    │
│   display/x11.rs       ARGB + Shape/XFixes                  │
│   display/wake.rs      Hyprland IPC + UNIX sock + SIGUSR1   │
│   sim.rs        icosahedron physics (SPEC.md)               │
│   ux.rs         VisibleIdle / Listening / Hidden            │
│   gaze.rs       look-target + dwell lock (pure)             │
│   camera.rs     V4L2 → gaze + latest RGB frame              │
│   radial.rs     icons + settings (UX / ID pages)            │
│   draw.rs       depth-sorted strokes + panel + auth chip    │
│   identity/     local multi-person gate (this doc)          │
└─────────────────────────────────────────────────────────────┘
```

## Identity

All matching is **local and fail-closed**. Nothing leaves the machine.

| Module | Always compiled | Optional backend |
| --- | --- | --- |
| `embed.rs` | L2 + cosine + gallery match | — |
| `gate.rs` | `Off / Face / Voice / Any / All` | — |
| `enroll.rs` | face / voice / gesture wizards | — |
| `persist.rs` | `~/.config/buckyboi/profiles.json` | — |
| `align.rs` | InsightFace `arcface_dst` similarity + warpAffine | — |
| `scrfd.rs` | letterbox / decode / NMS / largest-central pick | `face` → `ort` `det_10g.onnx` |
| `face.rs` | quality + detect → align → embed | `face` → ArcFace ONNX |
| `voice.rs` | log-mel 80-d speaker print + VAD-ish energy | `voice` → `cpal` + sherpa-onnx |
| `palm.rs` / `hands.rs` | 21-point rules + nearest-centroid | `hands` → palm + landmark ONNX |

Default `cargo run --release` is **gaze only** (V4L + overlay). Identity
state machines still compile and are unit-tested without models, camera,
or a microphone.

### Face

1. Detect (`det_10g.onnx` / SCRFD when `--features face` and the file
   exists): bbox + 5 landmarks (L-eye, R-eye, nose, L-mouth, R-mouth).
   Several faces → **largest / most central** (`area − 2·offset²`,
   InsightFace default). Skin-blob box if no detector.
2. **Align** with InsightFace `arcface_dst` (112×112 similarity /
   `norm_crop`). Bbox resize only if landmarks are missing.
3. Quality gates on the crop: area 4–85% of frame, brightness 40–230,
   Laplacian variance ≥ 12. Settings ID shows **NO FACE / TOO DARK /
   TOO BLURRY / TOO SMALL / TOO CLOSE / NO MODEL**.
4. Embed:
   - ArcFace `w600k_r50.onnx` / `w600k_mbf.onnx` → 512-d (`kind=arcface`),
     BGR `(x-127.5)/128`. Cosine defaults **0.35** (R50) / **0.40** (MBF).
   - else `BUCKYBOI_FACE_PROBE=1` → 32-d crop histogram (`kind=face-probe`).
     Prototype only — opt-in and weak.
   - else **no embedding** (fail closed).
5. Gaze look-target uses the SCRFD box when a detection is fresh;
   otherwise the skin/iris proxy. Vision ONNX is throttled to ~12.5 Hz.

Enrollment captures **8** accepted frames, then replaces that person’s face gallery.

### Voice

1. Mono 16 kHz samples (`cpal` when `--features voice`, or `BUCKYBOI_VOICE_SIM`).
2. Quality: ≥ 1.2 s and RMS energy ≥ 0.012. Rejects surface as
   **TOO SHORT** / **TOO QUIET**.
3. Embed: sherpa-onnx if an English CampPlus / ERes2Net file is present
   (`kind=sherpa`, cosine **≈ 0.60**); else 40-band log-mel mean+std
   (`kind=logmel`). `voice-sherpa` is an alias for `voice`.
4. Cosine vs stored prints.

Enrollment wants **3** utterances.

### Gestures

MediaPipe-style 21 landmarks. With `--features hands` and the OpenCV zoo
/ PINTO ONNX files, landmarks come from **palm detection → rotated ROI
→ hand landmark**. Rules + per-person centroids overlay that skeleton.
Without models, `BUCKYBOI_HAND_SIM` still drives the classifier.

Default map (overridable per person):

| Gesture | Action |
| --- | --- |
| fist | dismiss / hide |
| palm | listen |
| thumbs-up | confirm |
| point | info |
| peace | toggle settings |

Gestures may require a prior face unlock (`gestures_need_face`, default on).

### Gating

`AuthSession` holds live face and voice hits with an 8 s hold.

| Gate | Listen allowed when |
| --- | --- |
| `off` | always (fresh install) |
| `face` | live face match |
| `voice` | live voice match |
| `any` | face **or** voice |
| `all` | face **and** voice, same person |

Unknown never drives listen or sensitive gestures when a gate is on.

## Config

```
~/.config/buckyboi/
  settings.ini      look sliders + gate
  profiles.json     people, embeddings, gesture samples
  models/           optional ONNX (see scripts/download-models.sh)
```

Override with `BUCKYBOI_CONFIG` and `BUCKYBOI_MODELS`. `BUDDY_*` env aliases still work.

## Feature flags

```
cargo test --manifest-path rust/Cargo.toml
cargo test --manifest-path rust/Cargo.toml --no-default-features
cargo run --release --manifest-path rust/Cargo.toml
cargo run --release --manifest-path rust/Cargo.toml --features face,hands,voice
# `voice` now includes sherpa-onnx; log-mel is used if no speaker ONNX is present
```

Thin builds must keep working without models. ONNX crates are optional
and pull `ort` binaries only when those features are requested.

## Display

Runtime pick (`display::select_backend_kind`):

1. `BUCKYBOI_DISPLAY=wayland|x11|auto` (optional force).
2. If `WAYLAND_DISPLAY` is set, bind `zwlr_layer_shell_v1` (overlay
   layer, all anchors, exclusive zone 0, namespace `buckyboi`). Software
   pixels go through `wl_shm` (`Argb8888`, same BGRA layout as X11
   `PutImage`). Input: `wl_surface.set_input_region` = buddy disks.
3. If layer-shell is missing and `DISPLAY` is set, fall back to X11
   Shape + XFixes.
4. Else X11 only.

Both backends always compile so CI covers them. Multi-output: one
layer on the compositor-default / focused head.

### Wayland wake

There is no portable “any key / mouse anywhere” protocol. On
Hyprland / Omarchy the process polls `cursorpos` on the IPC socket and
listens on `$XDG_RUNTIME_DIR/buckyboi.sock` (`buckyboi --wake` /
`SUPER+B`). SIGUSR1 is the same wake. Documented in
`contrib/omarchy/`. Other wlroots compositors get the socket + signal
path; mouse-avoid while the pointer is *not* over a hit disk needs a
compositor cursor query (Hyprland only today).
