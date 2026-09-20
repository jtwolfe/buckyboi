# buckyboi (Linux x86_64)

This archive is `cargo build --release` with **default features**
(overlay + webcam gaze). Wayland (wlr-layer-shell) and X11 backends
are both compiled in.

## Not included

- InsightFace / sherpa-onnx / MediaPipe **ONNX weights**. Download them
  yourself (`scripts/download-models.sh` in this archive). Licenses are
  upstream; they are not MIT and are not vendored.
- `--features face,hands,voice` (ONNX Runtime + sherpa). Rebuild from
  source if you want those backends:
  `cargo build --release --manifest-path rust/Cargo.toml --features face,hands,voice`

## Run

```bash
chmod +x buckyboi
./buckyboi --help
# WAYLAND_DISPLAY or DISPLAY must be set
./buckyboi
```

Put `buckyboi` on `PATH` (e.g. `~/.local/bin`) if you want Hyprland
`exec-once = buckyboi`.

## Hyprland / Omarchy

See `contrib/omarchy/hyprland.conf`. **Source or copy the binds** into
your existing config. Do not replace `~/.config/hypr/hyprland.conf`.

```
exec-once = buckyboi
bind = SUPER, B, exec, buckyboi --wake
bind = SUPER, Escape, exec, buckyboi --quit
```

## Models (optional)

```bash
chmod +x scripts/download-models.sh
./scripts/download-models.sh
# or: BUCKYBOI_MODELS=/path/to/onnx ./scripts/download-models.sh
```
