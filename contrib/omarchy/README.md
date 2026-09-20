# buckyboi on Omarchy (Arch + Hyprland + Quickshell)

buckyboi stays a **standalone Rust binary**. It is not a Quickshell QML
plugin. On Omarchy it talks to Hyprland through **wlr-layer-shell**
(`zwlr_layer_shell_v1`) plus the Hyprland IPC socket.

## Install

```bash
sudo pacman -S --needed rust clang pkgconf libx11 libxfixes libxkbcommon wayland

git clone https://github.com/jtwolfe/buckyboi
cd buckyboi
cargo test --manifest-path rust/Cargo.toml
cargo run --release --manifest-path rust/Cargo.toml
# install the binary somewhere on PATH if you want exec-once:
# cargo install --path rust
```

Packaged install (AUR PKGBUILD in-tree, or a GitHub Release `.deb` /
tarball): see the **Install (packages)** section in the root README.
The Hyprland snippet is also installed to `/usr/share/buckyboi/` —
do not let a package overwrite `~/.config/hypr/hyprland.conf`.

Needs Rust **1.88+** (`rust/rust-toolchain.toml`).

## Hyprland

Copy or `source` [hyprland.conf](hyprland.conf):

```
source = ~/src/buckyboi/contrib/omarchy/hyprland.conf
```

That adds:

| Bind | Action |
| --- | --- |
| `exec-once = buckyboi` | start the overlay |
| `SUPER+B` | `buckyboi --wake` (hidden → visible) |
| `SUPER+Escape` | `buckyboi --quit` |
| `layerrule` | no animation / blur on namespace `buckyboi` |

## Wake story (honest)

| Event | Works on Hyprland / Omarchy? |
| --- | --- |
| Mouse move while hidden | **Yes** — the process polls `cursorpos` on `$XDG_RUNTIME_DIR/hypr/$HYPRLAND_INSTANCE_SIGNATURE/.socket.sock`. Does not steal focus. |
| `SUPER+B` / `buckyboi --wake` | **Yes** — writes `wake` to `$XDG_RUNTIME_DIR/buckyboi.sock` |
| `pkill -USR1 buckyboi` | **Yes** — SIGUSR1 is wake |
| Any key anywhere | **No**, not without a compositor bind. Wayland clients cannot see keys that are not sent to them. Bind extra keys to `buckyboi --wake` if you want them. |

Esc quits when the overlay has received OnDemand keyboard focus (click the
buddy). Use `SUPER+Escape` otherwise.

## Multi-monitor

v1 covers the **compositor-default / focused** output (layer-shell
`output = None`). Extra heads are a follow-up.

## Quickshell

You can launch buckyboi from a Quickshell IPC/shortcut the same way
Hyprland does (`buckyboi --wake`). Do not embed it as a QML item.
