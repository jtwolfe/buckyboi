#!/usr/bin/env bash
# Build the Linux x86_64 tarball and .deb from an already-built binary.
# Does not vendor ONNX models.
#
#   scripts/package-linux.sh <version> <binary> <output-dir>
#
# Example:
#   cargo build --release --manifest-path rust/Cargo.toml
#   scripts/package-linux.sh 0.5.0 rust/target/release/buckyboi dist

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <version> <binary> <output-dir>" >&2
  exit 2
fi

VERSION="$1"
BIN="$2"
OUT="$3"

if [[ ! -f "$BIN" ]]; then
  echo "missing binary: $BIN" >&2
  exit 1
fi
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-].*)?$ ]]; then
  echo "refusing version '$VERSION' (expected semver like 0.5.0)" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/buckyboi-pkg.XXXXXX")"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

NAME="buckyboi-${VERSION}-x86_64-linux"
TDIR="$STAGE/$NAME"
mkdir -p "$TDIR/contrib/omarchy" "$TDIR/scripts"

install -m0755 "$BIN" "$TDIR/buckyboi"
install -m0644 "$ROOT/packaging/linux/README.md" "$TDIR/README.md"
install -m0644 "$ROOT/LICENSE" "$TDIR/LICENSE"
install -m0644 "$ROOT/contrib/omarchy/hyprland.conf" "$TDIR/contrib/omarchy/hyprland.conf"
install -m0755 "$ROOT/scripts/download-models.sh" "$TDIR/scripts/download-models.sh"

tar -C "$STAGE" -czf "$OUT/${NAME}.tar.gz" "$NAME"
echo "wrote $OUT/${NAME}.tar.gz"

# --- .deb (amd64) -----------------------------------------------------------
DEB_ROOT="$STAGE/deb"
mkdir -p \
  "$DEB_ROOT/DEBIAN" \
  "$DEB_ROOT/usr/bin" \
  "$DEB_ROOT/usr/share/applications" \
  "$DEB_ROOT/usr/share/man/man1" \
  "$DEB_ROOT/usr/share/doc/buckyboi" \
  "$DEB_ROOT/usr/share/buckyboi"

install -m0755 "$BIN" "$DEB_ROOT/usr/bin/buckyboi"
install -m0644 "$ROOT/packaging/linux/buckyboi.desktop" \
  "$DEB_ROOT/usr/share/applications/buckyboi.desktop"
install -m0644 "$ROOT/packaging/linux/buckyboi.1" \
  "$DEB_ROOT/usr/share/man/man1/buckyboi.1"
gzip -n -9 "$DEB_ROOT/usr/share/man/man1/buckyboi.1"

install -m0644 "$ROOT/LICENSE" "$DEB_ROOT/usr/share/doc/buckyboi/copyright"
install -m0644 "$ROOT/packaging/linux/README.md" "$DEB_ROOT/usr/share/doc/buckyboi/README.md"
install -m0644 "$ROOT/contrib/omarchy/hyprland.conf" \
  "$DEB_ROOT/usr/share/buckyboi/hyprland.conf"
install -m0755 "$ROOT/scripts/download-models.sh" \
  "$DEB_ROOT/usr/share/buckyboi/download-models.sh"

SIZE_KB="$(du -sk "$DEB_ROOT" | awk '{print $1}')"

cat > "$DEB_ROOT/DEBIAN/control" <<EOF
Package: buckyboi
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: amd64
Maintainer: buckyboi contributors <https://github.com/jtwolfe/buckyboi>
Homepage: https://github.com/jtwolfe/buckyboi
Installed-Size: ${SIZE_KB}
Depends: libc6, libgcc-s1, libwayland-client0, libxkbcommon0
Recommends: libx11-6, libxext6, libxfixes3
Suggests: pipewire | pulseaudio
Description: Linux desktop overlay buddy (gaze / local identity)
 Fullscreen click-through overlay (Wayland wlr-layer-shell or X11
 Shape/XFixes) with optional webcam gaze. Default-features build —
 face/hands/voice ONNX backends are not compiled in.
 .
 ONNX model weights are not included. After install run
 /usr/share/buckyboi/download-models.sh (upstream licenses apply).
 Hyprland snippet: /usr/share/buckyboi/hyprland.conf (source/merge;
 do not replace ~/.config/hypr/hyprland.conf).
EOF

DEB="buckyboi_${VERSION}_amd64.deb"
dpkg-deb --root-owner-group --build "$DEB_ROOT" "$OUT/$DEB"
echo "wrote $OUT/$DEB"
