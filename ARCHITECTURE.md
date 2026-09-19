# Architecture

buckyboi is a **Linux-desktop overlay buddy**. Rust is the product. The
Python and Bend trees are historical ports of the original 800×600 window
toy and are not on the overlay / identity path.

```
┌─────────────────────────────────────────────────────────────┐
│  X11 overlay (bin: buckyboi)                                │
│  ARGB + Shape/XFixes click-through, Esc quits               │
│                                                             │
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
| `face.rs` | quality (area, brightness, Laplacian) + optional probe print | `face` → `ort` ArcFace ONNX |
| `voice.rs` | log-mel 80-d speaker print + VAD-ish energy | `voice` → `cpal`; `voice-sherpa` |
| `hands.rs` | 21-point rules + nearest-centroid calibration | `hands` → palm/landmark ONNX |

Default `cargo run --release` is **gaze only** (V4L + overlay). Identity
state machines still compile and are unit-tested without models, camera,
or a microphone.

### Face

1. Detect a face (skin-blob used by gaze; SCRFD/det ONNX when you drop one in).
2. Quality gates: area 4–85% of frame, brightness 40–230, Laplacian variance ≥ 12.
3. Embed:
   - `face` feature + `w600k_r50.onnx` / `w600k_mbf.onnx` → 512-d ArcFace (`kind=arcface`).
   - else `BUCKYBOI_FACE_PROBE=1` → 32-d crop histogram (`kind=face-probe`). Prototype only.
   - else **no embedding** (fail closed).
4. Cosine vs stored vectors. Below threshold → unknown.

Enrollment captures **8** accepted frames, then replaces that person’s face gallery.

### Voice

1. Mono 16 kHz samples (`cpal` when `--features voice`, or `BUCKYBOI_VOICE_SIM`).
2. Quality: ≥ 1.2 s and RMS energy ≥ 0.012.
3. Embed: 40-band log-mel mean+std (80-d, `kind=logmel`), or sherpa-onnx speaker ONNX (`kind=sherpa`).
4. Cosine vs stored prints.

Enrollment wants **3** utterances.

### Gestures

MediaPipe-style 21 landmarks. Rules recognize fist / palm / thumbs-up / point / peace.
Calibration stores per-class centroids (nearest-centroid / 1-NN). That is the
trainable path without shipping an extra MLP ONNX; you can still drop
PINTO/MediaPipe ONNX files for landmark extraction (`--features hands`).

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
```

Thin builds must keep working without models. ONNX crates are optional
and pull `ort` binaries only when those features are requested.

## Display

X11 (Shape + XFixes + 32-bit ARGB) is required for the overlay and for
global wake. XWayland usually works. Native Wayland is refused at
startup — there is no portable “any key / mouse anywhere” wake.
