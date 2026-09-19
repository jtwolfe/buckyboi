"""Headless checks for the shared simulation (no window)."""

from __future__ import annotations

import math
import unittest

from sim import (
    BOUND_RADIUS,
    EDGES,
    HI_X,
    HI_Y,
    HIT_RADIUS,
    IDLE_LOCKING,
    LO_X,
    LO_Y,
    LOCKED,
    MARGIN,
    SEEKING_LOCK,
    VERTS,
    WIN_H,
    WIN_W,
    corner,
    initial,
    other_corner,
    projected_edges,
    step,
)


class Geometry(unittest.TestCase):
    def test_twelve_unit_vertices(self) -> None:
        self.assertEqual(len(VERTS), 12)
        for x, y, z in VERTS:
            self.assertAlmostEqual(math.sqrt(x * x + y * y + z * z), 1.0, places=6)

    def test_thirty_edges(self) -> None:
        self.assertEqual(len(EDGES), 30)
        self.assertEqual(len(set(EDGES)), 30)


class Corners(unittest.TestCase):
    def test_other_corner_never_same(self) -> None:
        for lock in range(4):
            for mx, my in ((0, 0), (800, 0), (800, 600), (0, 600), (400, 300)):
                got = other_corner(lock, mx, my)
                self.assertNotEqual(got, lock)
                self.assertIn(got, range(4))

    def test_other_corner_flees_cursor(self) -> None:
        # Cursor in the top-left: farthest exclusive of lock 0 is BR (2).
        self.assertEqual(other_corner(0, 0.0, 0.0), 2)
        # Cursor in the bottom-right: farthest exclusive of lock 2 is TL (0).
        self.assertEqual(other_corner(2, 800.0, 600.0), 0)

    def test_lock_box_keeps_wireframe_on_screen(self) -> None:
        self.assertGreaterEqual(LO_X - BOUND_RADIUS, MARGIN)
        self.assertLessEqual(HI_X + BOUND_RADIUS, WIN_W - MARGIN)
        self.assertGreaterEqual(LO_Y - BOUND_RADIUS, MARGIN)
        self.assertLessEqual(HI_Y + BOUND_RADIUS, WIN_H - MARGIN)


class Motion(unittest.TestCase):
    def test_idles_into_home_corner(self) -> None:
        s = initial()
        self.assertEqual(s.mode, IDLE_LOCKING)
        self.assertEqual(s.lock, 0)
        for _ in range(400):
            s = step(s, -1000.0, -1000.0, False)
        tx, ty = corner(0)
        self.assertEqual(s.mode, LOCKED)
        self.assertLess(math.hypot(s.cx - tx, s.cy - ty), 2.5)
        self.assertTrue(LO_X <= s.cx <= HI_X)
        self.assertTrue(LO_Y <= s.cy <= HI_Y)

    def test_clamp_never_leaves_box(self) -> None:
        s = initial()
        s.cx, s.cy = 10.0, 10.0
        s.vx, s.vy = -40.0, -40.0
        s = step(s, -1000.0, -1000.0, False)
        self.assertGreaterEqual(s.cx, LO_X)
        self.assertGreaterEqual(s.cy, LO_Y)
        self.assertLessEqual(s.cx, HI_X)
        self.assertLessEqual(s.cy, HI_Y)

    def test_hover_retargets_then_seeks(self) -> None:
        s = initial()
        for _ in range(400):
            s = step(s, -1000.0, -1000.0, False)
        self.assertEqual(s.mode, LOCKED)
        self.assertEqual(s.lock, 0)
        # Sit the cursor on the body.
        s = step(s, s.cx, s.cy, False)
        self.assertNotEqual(s.lock, 0)
        self.assertIn(s.mode, (SEEKING_LOCK, 2))  # Floating or already seeking
        # Park the cursor far away so the hover-float can expire and seek.
        for _ in range(80):
            s = step(s, -1000.0, -1000.0, False)
        self.assertIn(s.mode, (SEEKING_LOCK, LOCKED, IDLE_LOCKING))
        self.assertNotEqual(s.lock, 0)

    def test_drag_follows_then_releases_to_other_corner(self) -> None:
        s = initial()
        for _ in range(400):
            s = step(s, -1000.0, -1000.0, False)
        ox, oy = s.cx, s.cy
        s = step(s, ox, oy, True)
        self.assertTrue(s.grabbed)
        s = step(s, 400.0, 300.0, True)
        self.assertAlmostEqual(s.cx, 400.0, delta=1.0)
        self.assertAlmostEqual(s.cy, 300.0, delta=1.0)
        prev_lock = s.lock
        s = step(s, 400.0, 300.0, False)
        self.assertFalse(s.grabbed)
        self.assertEqual(s.mode, SEEKING_LOCK)
        self.assertNotEqual(s.lock, prev_lock)

    def test_always_rotates(self) -> None:
        s = initial()
        s = step(s, -1000.0, -1000.0, False)
        self.assertGreater(s.ax, 0.0)
        self.assertGreater(s.ay, 0.0)

    def test_projected_edge_count(self) -> None:
        s = initial()
        edges = projected_edges(s)
        self.assertEqual(len(edges), 30)

    def test_repulsion_pushes_away(self) -> None:
        s = initial()
        for _ in range(400):
            s = step(s, -1000.0, -1000.0, False)
        cx0, cy0 = s.cx, s.cy
        # Approach from the right, just outside the hit disk.
        mx = cx0 + HIT_RADIUS + 20.0
        my = cy0
        s = step(s, mx, my, False)
        self.assertLess(s.cx, cx0)  # drifted left, away from the cursor


if __name__ == "__main__":
    unittest.main()
