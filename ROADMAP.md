# Roadmap

Rust overlay is the product. Phases below are implementation order, not
calendar estimates.

## A — Rebrand + baseline (done)

- Seed overlay (wireframe, gaze, radial settings, hide/wake) renamed to **buckyboi**.
- Crate / bin `buckyboi`, config `~/.config/buckyboi/`, `BUCKYBOI_*` env (with `BUDDY_*` aliases).
- Python / Bend kept as historical ports.

## B — Face enroll / verify (done, prototype-honest)

- Quality-checked capture, cosine gallery, fail-closed when no model.
- Settings → ID → FACE. `ort` ArcFace behind `--features face`.
- `BUCKYBOI_FACE_PROBE=1` weak histogram print for model-less machines.
- `BUCKYBOI_FACE_SIM=1` synthetic frames for VMs without a camera.

## C — Voice enroll / verify (done, prototype-honest)

- Log-mel speaker print always compiled; `cpal` + sherpa-onnx behind `--features voice`.
- English CampPlus / ERes2Net is the download default; ZH is opt-in.
- `BUCKYBOI_VOICE_SIM=1` tone path for headless / no-mic hosts.

## D — Gestures + calibration (done, prototype-honest)

- Rule classifier on 21 landmarks + per-person centroid training.
- Guided holds: fist → palm → thumbs-up → point → peace.
- Optional ONNX palm/landmark (`--features hands`).
- `BUCKYBOI_HAND_SIM=fist|palm|thumb|point|peace` on VMs.

## E — Multi-person + polish (this tree)

- Many people in `profiles.json`; re-enroll / delete; auth chip.
- Gate: off / face / voice / any / all; gestures may need prior face.
- Docs: README, UX, ARCHITECTURE, model download script.

## F — Wayland-native overlay (this tree)

- `wlr-layer-shell` overlay + `set_input_region` click-through on Hyprland / Omarchy / other wlroots.
- X11 path kept (feature-detect + `BUCKYBOI_DISPLAY`).
- Hidden wake: Hyprland `cursorpos` IPC, `$XDG_RUNTIME_DIR/buckyboi.sock`, SIGUSR1, `SUPER+B`.

## G — Biometric quality (this tree)

- SCRFD `det_10g` → 5-point `norm_crop` → ArcFace (bbox crop only if no landmarks).
- Gaze look-target from the SCRFD box when a detection is fresh.
- English sherpa speaker default; log-mel fallback; `voice` folds sherpa-onnx.
- Real palm + landmark ONNX when `--features hands` and models exist.
- Vision ONNX throttled to ~12.5 Hz; Settings ID shows TOO DARK / BLURRY / …

## Next (not blocking)

- First-run wizard that opens before the first listen when gate ≠ off and nobody is enrolled.
- Multi-output follow (one layer per head, or follow the focused output).
- Fractional-scale / `wp_viewporter` so HiDPI is sharp (v1 uses logical pixels).
- Packaged `.deb` / Flathub with a models extra.

## Honest limits

- Wayland overlay is **wlroots / Hyprland layer-shell**. GNOME Mutter
  without that protocol is X11-fallback or refuse.
- Hidden wake on Wayland is Hyprland IPC + a hotkey / socket — not
  “any key anywhere”.
- Face / voice prints here are **not** a commercial biometric. Thresholds fail closed; expect false rejects in bad light / noise.
- Cloud agent VMs usually have **no camera and no mic**. Use the `*_SIM` env vars and unit tests.
- `ort` / `sherpa-onnx` add compile time and download native libs — keep them off default features.
