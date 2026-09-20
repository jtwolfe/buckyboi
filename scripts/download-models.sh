#!/usr/bin/env bash
# Download optional offline ONNX models into ~/.config/buckyboi/models
# (or $BUCKYBOI_MODELS). Nothing here is required for `cargo test` or a
# gaze-only overlay.
#
# Weights keep their upstream licenses — download them yourself:
#   Face detector + ArcFace: InsightFace (buffalo_l / buffalo_sc).
#     Research / non-commercial terms often apply; read the InsightFace
#     license before shipping a binary that bundles these files.
#   Speaker: 3D-Speaker / sherpa-onnx release (per-model license).
#   Hands: MediaPipe (Apache-2.0) via OpenCV zoo ONNX conversions.
#
# Env:
#   BUCKYBOI_MODELS_ZH=1       also fetch the Chinese ERes2Net speaker net
#   BUCKYBOI_SKIP_HANDS=1      skip palm / landmark (they are optional)
#   BUCKYBOI_FACE_R50=1        also fetch w600k_r50.onnx (~174MB). Default rec is mbf.
#   BUCKYBOI_SKIP_GAZE_MESH=1  skip 478-pt face landmarker (iris gaze)

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

gh_release() {
  local url="$1"
  local out="$DEST/$2"
  if [[ -f "$out" ]]; then
    echo "have $out"
    return 0
  fi
  echo "GET $url"
  curl -fL --retry 3 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

# curl to .part, sha256, then mv. Mismatch → delete .part, do not install.
gh_release_sha() {
  local url="$1"
  local out="$DEST/$2"
  local want="$3"
  if [[ -f "$out" ]]; then
    echo "have $out"
    return 0
  fi
  echo "GET $url"
  curl -fL --retry 3 -o "$out.part" "$url" || {
    rm -f "$out.part"
    return 1
  }
  local got
  got="$(sha256sum "$out.part" | awk '{print $1}')"
  if [[ "$got" != "$want" ]]; then
    echo "warn: sha256 mismatch for $2 (got $got want $want) — not installing"
    rm -f "$out.part"
    return 1
  fi
  mv "$out.part" "$out"
}

echo "Models → $DEST"
echo
echo "Face — InsightFace buffalo (SCRFD det_10g + ArcFace). License: InsightFace."
echo "  Pipeline: SCRFD landmarks → 5-point norm_crop 112×112 → ArcFace."
echo "  Default rec net is MobileFaceNet (w600k_mbf). Existing kind=arcface"
echo "  galleries keep r50. BUCKYBOI_FACE_R50=1 also fetches w600k_r50 (~174MB)."
hf "public-data/insightface" "models/buffalo_l/det_10g.onnx" "det_10g.onnx" || \
  echo "warn: det_10g.onnx missing — gaze/ID fall back to the skin blob (no align)"
hf "public-data/insightface" "models/buffalo_sc/w600k_mbf.onnx" "w600k_mbf.onnx" || \
  echo "warn: w600k_mbf.onnx missing — trying r50"
if [[ "${BUCKYBOI_FACE_R50:-0}" != "0" ]] || [[ ! -f "$DEST/w600k_mbf.onnx" ]]; then
  hf "public-data/insightface" "models/buffalo_l/w600k_r50.onnx" "w600k_r50.onnx" || \
    echo "warn: w600k_r50.onnx (~174MB) missing"
fi
if [[ ! -f "$DEST/w600k_mbf.onnx" ]] && [[ ! -f "$DEST/w600k_r50.onnx" ]]; then
  echo "warn: no ArcFace rec net — face ID fails closed unless BUCKYBOI_FACE_PROBE=1"
fi

echo
echo "Speaker — English 3D-Speaker via sherpa-onnx (default). Log-mel is fallback."
SPK_BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"
gh_release \
  "${SPK_BASE}/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx" \
  "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx" \
  || gh_release \
       "${SPK_BASE}/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx" \
       "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx" \
  || echo "warn: English speaker model failed — log-mel print still works"

if [[ "${BUCKYBOI_MODELS_ZH:-0}" != "0" ]]; then
  echo "ZH speaker (optional, BUCKYBOI_MODELS_ZH=1)"
  gh_release \
    "${SPK_BASE}/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
    || echo "warn: ZH ERes2Net missing"
fi

echo
echo "Hands — MediaPipe palm + landmark (OpenCV zoo ONNX, Apache-2.0)."
if [[ "${BUCKYBOI_SKIP_HANDS:-0}" == "0" ]]; then
  hf "opencv/palm_detection_mediapipe" \
    "palm_detection_mediapipe_2023feb.onnx" "palm_detection.onnx" \
    || echo "warn: palm_detection.onnx missing — drop a 192×192 MediaPipe/PINTO palm ONNX here"
  hf "opencv/handpose_estimation_mediapipe" \
    "handpose_estimation_mediapipe_2023feb.onnx" "hand_landmark.onnx" \
    || echo "warn: hand_landmark.onnx missing — drop a 224×224 landmark ONNX here"
else
  echo "skip hands (BUCKYBOI_SKIP_HANDS=1)"
fi

echo
echo "Gaze — MediaPipe Face Landmarker 478-pt iris (yakhyo ONNX, Apache-2.0)."
if [[ "${BUCKYBOI_SKIP_GAZE_MESH:-0}" == "0" ]]; then
  gh_release_sha \
    "https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/download/weights/face_landmarker_Nx3x256x256.onnx" \
    "face_landmarker.onnx" \
    "111795f8703cdeb6d0c68a9f3cc966a0f23f8786bb00f4577a11f461fc4276ac" \
    || echo "warn: face_landmarker.onnx missing — gaze stays face-box / skin proxy"
else
  echo "skip gaze mesh (BUCKYBOI_SKIP_GAZE_MESH=1)"
fi

echo
echo "Done. Point the overlay at this directory with BUCKYBOI_MODELS=$DEST"
echo "Build: cargo run --release --features face,hands,voice"
echo
echo "Thresholds (cosine, L2-normalized; override in profiles.json or env):"
echo "  face  w600k_mbf ≈ 0.40   w600k_r50 ≈ 0.35   (BUCKYBOI_FACE_THRESHOLD)"
echo "  rec   empty/mbf gallery → mbf; kind=arcface gallery → r50"
echo "        override: BUCKYBOI_FACE_REC=mbf|r50|/path"
echo "  voice sherpa ≈ 0.60      log-mel ≈ 0.62     (BUCKYBOI_VOICE_THRESHOLD)"
echo "  enroll voice: 3 utterances ≥ 1.2 s"
echo "Without models, face ID fails closed unless BUCKYBOI_FACE_PROBE=1 (weak histogram)."
