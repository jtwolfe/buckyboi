#!/usr/bin/env bash
# Download optional offline ONNX models into ~/.config/buckyboi/models
# (or $BUCKYBOI_MODELS). Nothing here is required for `cargo test` or a
# gaze-only overlay. Face / sherpa speaker / hand models are large.

set -euo pipefail

DEST="${BUCKYBOI_MODELS:-${HOME}/.config/buckyboi/models}"
mkdir -p "$DEST"

hf() {
  local repo="$1"
  local file="$2"
  local out="$DEST/$3"
  if [[ -f "$out" ]]; then
    echo "have $out"
    return 0
  fi
  local url="https://huggingface.co/${repo}/resolve/main/${file}"
  echo "GET $url"
  curl -fL --retry 3 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

echo "Models → $DEST"
echo
echo "Face (InsightFace buffalo_l / buffalo_sc via public-data mirror)"
hf "public-data/insightface" "models/buffalo_sc/w600k_mbf.onnx" "w600k_mbf.onnx" || \
  echo "warn: buffalo_sc recognition missing — try buffalo_l w600k_r50.onnx (~174MB)"
hf "public-data/insightface" "models/buffalo_l/w600k_r50.onnx" "w600k_r50.onnx" || true
hf "public-data/insightface" "models/buffalo_l/det_10g.onnx" "det_10g.onnx" || true

echo
echo "Speaker (sherpa-onnx 3D-Speaker ERes2Net, optional — needs --features voice-sherpa)"
SPK_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
if [[ ! -f "$DEST/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" ]]; then
  curl -fL --retry 3 -o "$DEST/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx.part" "$SPK_URL" \
    && mv "$DEST/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx.part" \
          "$DEST/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
    || echo "warn: speaker model download failed (log-mel print still works)"
else
  echo "have speaker model"
fi

echo
echo "Hands (PINTO MediaPipe-style palm / landmark — optional)"
# Community mirrors; skip quietly if they move.
hf "qualcomm/MediaPipe-Hand-Detection" "Hand_Landmark.onnx" "hand_landmark.onnx" || \
  echo "warn: drop a palm/landmark ONNX named palm_detection.onnx or hand_landmark.onnx into $DEST"

echo
echo "Done. Point the overlay at this directory with BUCKYBOI_MODELS=$DEST"
echo "Build: cargo run --release --features face,hands,voice"
echo "Without models, face ID fails closed unless BUCKYBOI_FACE_PROBE=1 (weak histogram print)."
