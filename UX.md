# Overlay UX

Linux desktop buddy. Approximate **webcam gaze** is a look-target, not
research-grade eye tracking. Face / voice / gesture identity is **local**,
fail-closed, and prototype-grade — see [ARCHITECTURE.md](ARCHITECTURE.md).

## Phases

```
VisibleIdle ──gaze-follow 5–15s──► Listening ──timeout──► Hidden
     │              or click-in-place                         │
     │              (blocked if gate is on and auth=UNKNOWN)  │
     ▲                                                        │
     └──────── mouse move or key (first event) ───────────────┘
```

| Phase | What you see | Input |
| --- | --- | --- |
| `VisibleIdle` | Rotating wireframe. Soft-avoids **gaze** (or the mouse). Dwell ring. Auth chip (`ADA` / `UNKNOWN`). | Click-through except the buddy. Tap starts Listening *if the gate allows*. Drag relocates. |
| `Listening` | Brighter wire + pulse. Radial icons. Settings: **UX** and **ID** tabs. First-run / add-person wizard or Settings ID enroll panel. Gaze **CALIB** dots (fullscreen paint). | Icons + panel in the hit region. Timer **pauses** while Settings, **gaze calib**, or the first-run wizard is open. |
| `Hidden` | Overlay unmapped (X11) or transparent + empty `set_input_region` (Wayland) | **X11:** first root-pointer move (≥ 1 px) or keymap change. **Wayland / Hyprland:** cursorpos IPC, `buckyboi --wake`, SIGUSR1, or `SUPER+B`. |

Esc quits from any phase on X11 (keymap poll, no grab). On Wayland, Esc
works after the layer has OnDemand keyboard focus (click the buddy);
`SUPER+Escape` → `buckyboi --quit` always works. See
[contrib/omarchy/](contrib/omarchy/).

## Gaze (prototype)

This is a **laptop webcam look-target**, not a Tobii. After a 5-point
calib expect **~3–6° / buddy-hit** (`HIT_RADIUS` = 100 px), not
reading-gaze. Eyes moving with a still head should move the target.

Pipeline, all local / offline:

1. V4L2 capture (`/dev/video0` or `BUCKYBOI_CAMERA`) — prefer 640×480,
   else 320×240, YUYV or MJPEG.
2. If `--features face` and `face_landmarker.onnx` are present: SCRFD
   box (margin 0.25, optional 5-kps rotate) → 478-pt mesh → iris 468/473
   in the eye box + nose tip 1. The **vision worker** maps to screen
   (affine from `gaze_calib.json`, else uncalibrated 2D gains). Present
   only smooths.
3. Else if `det_10g.onnx`: SCRFD face box with `FACE_GAIN`. Skin-blob
   iris proxy is the last fallback.
4. Exponential smooth (`GAZE_SMOOTH` = 0.14). Hold last sample 280 ms if a frame misses.

RGB frames are also stashed for face / hand identity. ONNX identity runs
at ~12.5 Hz so the overlay can stay near 60 Hz. `BUCKYBOI_NO_CAMERA=1`
or a failed open **falls back** to mouse-avoid + click-to-listen.
`BUCKYBOI_GAZE_SIM=mouse|chase` bypasses the tracker.

### Gaze-follow lock

While `VisibleIdle` and a gaze sample is on the buddy (hysteresis: enter at
`HIT_RADIUS + 8`, leave at `HIT_RADIUS + 22`):

- the body **stops fleeing**
- a timer runs for a random **5–15 s** (`gaze_lock_ms`)
- a 220 ms grace lets the estimate flicker
- when the timer fires → `Listening` **only if** `AuthSession` satisfies the gate

`BUCKYBOI_GAZE_SIM=mouse` treats the pointer as gaze.
`BUCKYBOI_GAZE_SIM=chase` eases a scripted look-target onto the buddy.

## Auth chip

Always painted above the body while mapped:

| Label | Meaning |
| --- | --- |
| person name | live face and/or voice match, still inside the 8 s hold |
| `UNKNOWN` | no live match (or hold expired) |

Info icon logs gate + auth + enrolled count.

## Gating

UX tab **GATE** cycles `OFF → FACE → VOICE → ANY → ALL`.

| Gate | Who may listen |
| --- | --- |
| OFF | anyone (default on a fresh install) |
| FACE | enrolled face match |
| VOICE | enrolled voice print |
| ANY | face or voice |
| ALL | same person on face **and** voice |

A blocked click or gaze-lock logs `listen blocked` / `gaze-lock ignored`
and stays in `VisibleIdle`. Gestures that start listen use the same check.
**HANDS NEED FACE** (ID tab) requires a live face unlock before fist/palm/…

## First run

If **GATE ≠ OFF** and `profiles.json` has **nobody**, the overlay opens a
guided panel on the buddy (same chrome as Settings) instead of leaving
you stuck on `UNKNOWN` / Settings → ID:

1. **Name** — default `P1` (NEXT). Rename later in `profiles.json` if you want.
2. **FACE** — quality frames with live rejects (`NO FACE`, `TOO DARK`,
   `TOO BLURRY`, …) and `3/8` progress. Several faces → largest / most
   central, chip **USING LARGEST FACE**.
3. **VOICE** — 3 utterances (`TOO SHORT` / `TOO QUIET`). Skip if no mic.
4. **HANDS** — optional fist → palm → thumb → point → peace. Skip allowed.
   Missing palm/landmark ONNX → **NO MODEL** (wizard skips remaining
   holds). No palm in frame → **NO HAND**; ~6.4 s of continuous miss
   skips so the wizard can continue. A fist after a flicker still enrolls.
5. **Done** — if GATE is still OFF, suggest **FACE** (or **ANY** when
   voice was saved too). Auth chip shows the name. **ADD** starts
   another person.

Esc **cancels the wizard** and leaves the overlay up (press Esc again to
quit). Settings → **ID** → **ADD** reuses this wizard. FACE / VOICE /
HAND on that tab still enroll one modality at a time via the same
machines.

The wizard also opens when GATE is turned on with an empty gallery, or
on the first blocked listen / gaze-lock that needs a person.

## Calibration / enrollment

Settings → **ID**:

1. **NEW** creates `P1`, `P2`, … (rename later by editing `profiles.json` if you want a real name).
2. Select a row.
3. **FACE** — “LOOK AT CAM”. Eight frames must pass area / brightness / sharpness.
   Rejects show as **NO FACE / TOO DARK / TOO BLURRY / TOO SMALL /
   TOO CLOSE / NO MODEL**. Missing ArcFace model → no save unless
   `BUCKYBOI_FACE_PROBE=1`.
4. **VOICE** — “SAY PHRASE”. Three clips ≥ 1.2 s with enough energy
   (**TOO SHORT** / **TOO QUIET**). Prefer the English sherpa model
   when present. No mic → `BUCKYBOI_VOICE_SIM=1` (tone) for bringing up the wizard.
5. **HAND** — guided holds, starting at **FIST**. HUD **HOLD FIST** then
   **FIST n/6** (16-char chip). Missing palm/landmark ONNX → **NO MODEL**
   (fails immediately; the wizard skips hands). No palm in frame →
   **NO HAND**; ~6.4 s of continuous miss fails so the wizard can skip.
   A fist after a flicker still enrolls (`Ok` zeroes the reject cap).
   Use `BUCKYBOI_HAND_SIM=fist` (then `palm`, `thumb`, `point`, `peace`)
   without a camera.
6. **DEL** removes that person. **CANCEL** aborts a running Settings enroll.
7. **ADD** opens the same first-run wizard for another person.

Re-enroll replaces that modality. Profiles: `~/.config/buckyboi/profiles.json`.

### Gesture map (default)

| Hold | Buddy action |
| --- | --- |
| fist | hide |
| palm | listen (if gate allows) |
| thumbs-up | confirm (reserved) |
| point | info log |
| peace | toggle settings |

Custom samples become nearest-centroid features per person. Rule
classifier is the fallback when a class has no samples yet.

## Click vs drag

Press+release on the hit disk is a **tap** when movement `< 6 px` and
duration `< 250 ms`. Crossing 6 px while held is a **drag**. Hover-push
from the original window ports stays **off**.

## Physics (while visible)

`step_overlay_avoid`: always rotate; soft avoid from gaze when present;
no flee while the avoid point is on the hit disk; corner lock; drag after
the tap threshold; no hover-push. Listening uses `spin` only.

## Radial icons (Listening)

Four 17 px discs on a 122 px orbit:

| Icon | Action |
| --- | --- |
| **Settings** | Toggle the panel (UX / ID). |
| **Dismiss** | Hide immediately. |
| **Mute** | Local flag only (no ASR mute). |
| **Info** | Log auth / gate / people. |

### Settings — UX

- **CAM** — enable/disable V4L gaze
- **GATE** — cycle identity policy
- **LISTEN** / **GAZE** / **STROKE** sliders
- **CALIB** — 5-dot dwell (center, TL, TR, BR, BL, inset 10%). Stays
  Listening; hide timer pauses; SKIP / CANCEL chips near the buddy.
  Needs both irises on ≥ 8 snaps per point. Saves
  `~/.config/buckyboi/gaze_calib.json` unless RMSE is worse than
  `min(200 px, 15% of the short side)`. Screen-size mismatch → stale
  (`RECALIBRATE`); previous file kept on Esc / cancel.

Written to `~/.config/buckyboi/settings.ini`. Env overrides still win at startup.

## Listen timeout

Default from settings / `listen_duration_ms`. `BUCKYBOI_LISTEN_MS` overrides.
Hidden once the deadline hits **and** Settings / gaze calib / the
first-run wizard is closed.

## Wireframe strokes

Darker halo + brighter core. Edges sorted far→near (`draw::edges_far_to_near`).

## Click-through / wake

**X11:** fullscreen 32-bit ARGB. Shape + XFixes set both `Input` and
`Bounding`. Never `SetInputFocus`, never grab. Hidden: unmap + empty
shape; poll `XQueryPointer` and `XQueryKeymap`.

**Wayland:** wlr-layer-shell overlay, empty `wl_surface` input region
except buddy / icons / panel. Never grab. Hidden: transparent commit +
empty region. Wake is compositor-specific (Hyprland `cursorpos` +
control socket). True “any key anywhere” is not possible as a plain
Wayland client.
