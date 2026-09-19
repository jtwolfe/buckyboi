#!/usr/bin/env python3
"""Rotating wireframe icosahedron — Python port (pygame). See ../SPEC.md."""

from __future__ import annotations

import sys

import pygame

from sim import (
    BG,
    HIT_RADIUS,
    WIN_H,
    WIN_W,
    initial,
    mix_wire,
    projected_edges,
    step,
)


def _plot(surf: pygame.Surface, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIN_W and 0 <= y < WIN_H:
        surf.set_at((x, y), color)


def draw_line(
    surf: pygame.Surface,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rgb: tuple[int, int, int],
) -> None:
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        _plot(surf, x, y, rgb)
        _plot(surf, x + 1, y, rgb)
        _plot(surf, x, y + 1, rgb)


def rgb(color: int) -> tuple[int, int, int]:
    return ((color >> 16) & 255, (color >> 8) & 255, color & 255)


def main() -> int:
    pygame.init()
    pygame.display.set_caption("Icosahedron")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    state = initial()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        mx, my = pygame.mouse.get_pos()
        button = bool(pygame.mouse.get_pressed()[0])
        state = step(state, float(mx), float(my), button)

        screen.fill(rgb(BG))
        for ax, ay, bx, by, z in projected_edges(state):
            draw_line(screen, ax, ay, bx, by, rgb(mix_wire(z)))

        # Soft hit-disk hint when the cursor is over the body.
        if (mx - state.cx) ** 2 + (my - state.cy) ** 2 <= HIT_RADIUS * HIT_RADIUS:
            pygame.draw.circle(
                screen, (40, 44, 56), (int(state.cx), int(state.cy)), int(HIT_RADIUS), 1
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
