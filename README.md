# buckyboi

A Linux-desktop overlay **buddy**: a rotating wireframe icosahedron that
floats above your windows, lazily avoids your look, and can **identify
one or many people** by face, voice, or gesture — all offline.

Rust is the product (`rust/` crate and `buckyboi` binary). Python and
Bend are historical ports of the original 800×600 window toy.

```
cargo test --manifest-path rust/Cargo.toml
cargo run --release --manifest-path rust/Cargo.toml
```

`WAYLAND_DISPLAY` (Hyprland / wlroots) or `DISPLAY` (X11) must be set.
The binary is `buckyboi`. `buckyboi --help` prints usage. Feature flags
keep thin builds working without ONNX or a mic.

## What you get

1. **Overlay** — fullscreen transparent click-through window (native
   **Wayland** wlr-layer-shell on Hyprland / Omarchy, or **X11**
   Shape/XFixes), depth-sorted strokes, listening radial icons + settings,
   hide / wake, Esc quits.
2. **Gaze** — webcam look-target; lazy avoid; dwell → listen. Not a
   calibrated eye tracker. With `--features face` and `det_10g.onnx`,
   the look-target comes from the SCRFD face box (largest / most
   central if several people are in frame). Skin blob is the fallback.
3. **Multi-person identity** — enroll people under `~/.config/buckyboi/`.
   Cosine match, fail closed. No cloud.
4. **Gating** — only enrolled people can start listen / sensitive
   actions (configurable: face and/or voice; gestures may need a prior
   face unlock).

See [UX.md](UX.md) for the state machine, [ARCHITECTURE.md](ARCHITECTURE.md)
for the stack, [SPEC.md](SPEC.md) for icosahedron physics.

## Interaction

1. The buddy sits on a fullscreen overlay (home corner: top-left) and always rotates.
2. **Gaze** softly repels it. Stay on the body for a random **5–15 s**
   (`BUCKYBOI_GAZE_LOCK_MS`) → **Listening** (cyan pulse), *if the gate allows*.
3. **Click in place** still works as a fallback (press+release, move &lt; 6 px, &lt; 250 ms).
4. In **Listening**, a ring of icons orbits the body. **Settings** has two
   tabs: **UX** (camera / gate / listen / gaze / stroke) and **ID**
   (people, enroll face / voice / gestures, delete).
5. After a random **5–15 s** of Listening it **hides**. The timer pauses
   while Settings or the first-run enroll wizard is open.
6. While hidden, the first **mouse movement or key press** brings it back
   (X11). On Wayland / Hyprland: mouse via compositor IPC, or `SUPER+B`
   / `buckyboi --wake`.

An **auth chip** above the body shows the matched name or `UNKNOWN`.

Esc quits. Settings persist in `~/.config/buckyboi/settings.ini`.
People and embeddings persist in `~/.config/buckyboi/profiles.json`.

## Feature flags

| Feature | Default | What it adds |
| --- | --- | --- |
| `gaze` | yes | V4L2 webcam (`v4l` + `image`) |
| `face` | no | `ort` + SCRFD detect + 5-point align + ArcFace ONNX |
| `hands` | no | `ort` + MediaPipe palm / landmark ONNX |
| `voice` | no | `cpal` mic + sherpa-onnx speaker extract (if a model is present) |
| `voice-sherpa` | no | alias for `voice` |

```bash
# overlay + webcam gaze (no ONNX)
cargo run --release --manifest-path rust/Cargo.toml

# identity backends (models still optional at runtime)
cargo run --release --manifest-path rust/Cargo.toml --features face,hands,voice

# thinnest compile (no V4L)
cargo run --release --manifest-path rust/Cargo.toml --no-default-features
```

Gaze math, follow timer, enrollment machines, cosine match, log-mel
voice print, and gesture rules **always compile** and are unit-tested.

## Models (optional, first-run)

```bash
chmod +x scripts/download-models.sh
./scripts/download-models.sh
# or: BUCKYBOI_MODELS=/path/to/onnx ./scripts/download-models.sh
```

| File in `~/.config/buckyboi/models/` | Used by |
| --- | --- |
| `det_10g.onnx` | `--features face` SCRFD (bbox + 5 landmarks) |
| `w600k_r50.onnx` or `w600k_mbf.onnx` | `--features face` ArcFace after `norm_crop` |
| `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` | `--features voice` sherpa (English default) |
| `3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx` | EN fallback speaker net |
| `palm_detection.onnx` / `hand_landmark.onnx` | `--features hands` MediaPipe ONNX |

Face path when both detector and rec nets are present: **detect →
5-point similarity align (`arcface_dst` 112×112) → BGR `(x-127.5)/128`
→ ArcFace**. A bbox-only resize is the degraded fallback if landmarks
are missing. Cosine on L2-normalized embeddings; defaults **0.35**
(R50) / **0.40** (MBF). Override with `face_threshold` in
`profiles.json` or `BUCKYBOI_FACE_THRESHOLD`. Several faces: keep the
**largest / most central** (InsightFace `area − 2·offset²`).

InsightFace weights are **not** MIT. Read their license before
redistributing `det_10g` / `w600k_*`. They are not vendored.

Without a face rec model, enrollment **fails closed** unless you opt
into the weak probe print: `BUCKYBOI_FACE_PROBE=1`. That is a crop
histogram, not InsightFace.

Voice: `--features voice` prefers sherpa when an English CampPlus /
ERes2Net file is in the models dir (threshold **≈ 0.60**; enroll **3**
utterances ≥ 1.2 s). `BUCKYBOI_MODELS_ZH=1` also pulls the ZH net.
Log-mel is fallback only.

## First run

When **GATE ≠ OFF** and nobody is in `profiles.json`, a guided overlay
wizard opens on the buddy (Omarchy / day-one). Face first (live
`NO FACE` / `TOO DARK` / `TOO BLURRY` + `3/8`), then voice (skippable),
then optional hands. Esc cancels the wizard without quitting. **ADD**
on Settings → ID reuses it. If GATE is still OFF at Done, it suggests
FACE or ANY.

## Enrollment / calibration

Open Listening → Settings → **ID**:

| Control | Action |
| --- | --- |
| **ADD** | first-run wizard for a new person (`P1`, `P2`, …) |
| select a row | who the next single-modality enroll applies to |
| **FACE** | capture 8 quality-checked frames |
| **VOICE** | capture 3 utterances ≥ 1.2 s |
| **HAND** | guided holds starting at FIST (then palm, thumb, point, peace) |
| **DEL** | delete the selected person |
| **HANDS NEED FACE** | toggle “gestures require a live face unlock” |

Re-enroll replaces that modality for the selected person. Gate on the
**UX** tab: `OFF` → `FACE` → `VOICE` → `ANY` → `ALL`.

## Run (Omarchy / Hyprland / Wayland)

Needs Rust **1.88+** (`rust-toolchain.toml` pins 1.88 — `v4l` / `image`
need it). On **Omarchy** (Arch + Hyprland + Quickshell) and other
wlroots compositors the binary is a **standalone** overlay — not a
Quickshell plugin.

```bash
cd rust
cargo test          # compiles X11 + Wayland backends
cargo run --release # WAYLAND_DISPLAY → layer-shell; else X11
```

Drop in [contrib/omarchy/hyprland.conf](contrib/omarchy/hyprland.conf):

```
exec-once = buckyboi
bind = SUPER, B, exec, buckyboi --wake
bind = SUPER, Escape, exec, buckyboi --quit
```

`buckyboi --wake` / `--quit` talk to `$XDG_RUNTIME_DIR/buckyboi.sock`.
SIGUSR1 also wakes. See [contrib/omarchy/README.md](contrib/omarchy/README.md).

Override: `BUCKYBOI_DISPLAY=x11` (or `wayland`). If layer-shell bind
fails and `DISPLAY` is set, the process falls back to X11.

## Run (X11)

An X11 display with **Shape** and **XFixes**, and a 32-bit ARGB visual.
Webcam support needs a V4L2 device and the default `gaze` feature (clang
builds the V4L bindings once). Microphone needs `--features voice` plus
a PipeWire / Pulse / ALSA source.

```bash
cd rust
BUCKYBOI_DISPLAY=x11 cargo run --release
```

```bash
# shorter listen / gaze-lock
BUCKYBOI_LISTEN_MS=3000 BUCKYBOI_GAZE_LOCK_MS=3000 cargo run --release

# no webcam
BUCKYBOI_NO_CAMERA=1 cargo run --release

# pointer stands in for gaze
BUCKYBOI_GAZE_SIM=mouse BUCKYBOI_GAZE_LOCK_MS=4000 cargo run --release

# scripted look-at-buddy (agent VMs)
BUCKYBOI_GAZE_SIM=chase BUCKYBOI_GAZE_LOCK_MS=3500 cargo run --release

# identity without hardware (this VM has no /dev/video* and no mic)
BUCKYBOI_FACE_SIM=1 BUCKYBOI_FACE_PROBE=1 \
  BUCKYBOI_VOICE_SIM=1 BUCKYBOI_HAND_SIM=palm \
  cargo run --release
```

`BUDDY_*` names still work as aliases.

## Distro notes

| Need | Debian / Ubuntu | Fedora | Arch |
| --- | --- | --- | --- |
| Rust 1.88+ | `rustup` (toolchain file) | same | same |
| X11 + Shape + XFixes | `libx11-dev libxext-dev libxfixes-dev` | `libX11-devel libXfixes-devel` | `libx11 libxfixes` |
| Wayland + xkbcommon | `libwayland-dev libxkbcommon-dev` | `wayland-devel libxkbcommon-devel` | `wayland libxkbcommon` |
| clang (V4L bindgen) | `clang libclang-dev` | `clang` | `clang` |
| Camera | user in `video`, `/dev/video0` | same | same |
| Mic | PipeWire / Pulse; user in `audio` | same | same |

Wayland-only sessions (Hyprland, Sway, labwc, …): native layer-shell
overlay. GNOME / KWin without `zwlr_layer_shell_v1` fall back to X11
when `DISPLAY` is set, otherwise exit. There is no portable global-input
wake on native Wayland — Omarchy uses Hyprland IPC + a hotkey (below).

## Camera / mic fallbacks

| Situation | What happens |
| --- | --- |
| `/dev/video0` readable (or `BUCKYBOI_CAMERA`) | Capture thread; SCRFD box (if `face` + `det_10g`) or skin/iris proxy → look-target; RGB stashed for identity. ONNX identity is throttled to ~12.5 Hz |
| Device missing, busy, or `EACCES` | Log + mouse-avoid + click-to-listen. Overlay still runs |
| `BUCKYBOI_NO_CAMERA=1` | Skip V4L |
| No mic / no `voice` feature | Voice enroll waits for `BUCKYBOI_VOICE_SIM` or injected samples |
| Cloud agent VM | Typically **no camera, no mic**. Unit tests cover identity; use `*_SIM` |

**Honesty:** gaze is “where is the face in the frame?”, not research-grade
tracking. Face / voice prints here are **prototypes**. They fail closed
and will false-reject in bad light or noise. Do not use this as a lock
screen.

## Click-through

**X11:** 32-bit ARGB, override-redirect, stacked `Above`,
`_NET_WM_WINDOW_TYPE_DOCK`, Shape + XFixes input *and* bounding = buddy
disk + icons + panel. Empty region + `UnmapWindow` while hidden. No
pointer/keyboard grab; wake uses `XQueryPointer` + `XQueryKeymap`.
Never `SetInputFocus`.

**Wayland:** `zwlr_layer_shell_v1` **overlay** layer, anchored to all
edges, `exclusive_zone = 0`. `wl_surface.set_input_region` is empty
except the same buddy / icon / panel disks. Hidden commits a transparent
buffer with an empty region (no grab, no focus steal). Keyboard
interactivity is **OnDemand** so Esc works after you click the buddy.

## X11 vs Wayland

| Environment | Overlay | Click-through | Gaze | Wake |
| --- | --- | --- | --- | --- |
| **X11** | Fullscreen ARGB + Shape | Yes | V4L2 if present | Root pointer + keymap |
| **XWayland** | Same if `DISPLAY` is set | Usually | Same | X11 side |
| **Native Wayland** (Hyprland / Omarchy / other wlroots) | wlr-layer-shell overlay | `set_input_region` disks | Same | Hyprland `cursorpos` IPC + `buckyboi --wake` / SUPER+B. **Not** “any key anywhere”. |
| **Wayland without layer-shell** | Falls back to X11 if `DISPLAY` is set | — | — | — |

Multi-output: the layer is created with `output = None` (compositor
default / focused head). Extra monitors are a follow-up.

## Tests

```bash
cargo test --manifest-path rust/Cargo.toml
cargo test --manifest-path rust/Cargo.toml --no-default-features
```

Headless: vertex/edge math, lock/clamp, overlay avoid, tap vs drag,
listen timeout, gaze lock, face-blob + iris, radial hits, stroke
occlusion, **cosine match / fail-closed**, **gate state machine**,
**enroll machines**, **first-run wizard FSM**, **5-point align math**, **SCRFD decode**,
**log-mel + speaker-path pick**, **gesture rules + centroids**.

```bash
python3 python/test_sim.py          # historical window sim
# bend PROOF.bend                   # historical Bend laws
```

## Project layout

```
README.md          this file
ARCHITECTURE.md    crate map + identity stack
ROADMAP.md         phases and honest leftovers
UX.md              overlay + auth + calibration flows
SPEC.md            shared icosahedron physics
contrib/omarchy/   Hyprland binds + Omarchy notes
contrib/hyprland/  same snippet for generic Hyprland
scripts/           model download
rust/              product crate — lib + bin buckyboi
python/            historical pygame window
bend/              historical Bend 2 window
```

## License

MIT. Third-party ONNX weights keep their upstream licenses (InsightFace,
sherpa-onnx, MediaPipe / PINTO). Download them yourself; they are not
vendored.
