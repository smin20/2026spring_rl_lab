import os
from typing import Optional, Sequence, Union

import numpy as np
import pygame
import yaml


class GridWorldEnv_c2:
    """
    Continuous-action GridWorld.

    State is a continuous (row, col) position in meters.
    Action is a 2D vector in [-1, 1]^2, scaled by step_size_m.
    """

    def __init__(
        self,
        config_path: str,
        cell_size_px: int = 100,
        cell_size_m: float = 1.0,
        step_size_m: float = 1.0,
        agent_radius_px: Optional[int] = 20,
        headless: bool = False,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.width = cfg["width"]
        self.height = cfg["height"]
        raw_map = cfg["map"]
        if isinstance(raw_map[0], str):
            self.grid = np.array([[int(c) for c in row] for row in raw_map], dtype=int)
        else:
            self.grid = np.array(raw_map, dtype=int)

        if cfg.get("start") is not None:
            self.start_cell = tuple(cfg["start"])
        else:
            zeros = np.argwhere(self.grid == 0)
            if zeros.size == 0:
                raise ValueError("No empty cell (0) in map to start.")
            self.start_cell = tuple(zeros[0])

        self.cell_size_m = cell_size_m
        self.step_size_m = step_size_m

        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()

        self.cell_size_px = cell_size_px
        self.agent_radius = (
            int(cell_size_px * 0.4) if agent_radius_px is None else agent_radius_px
        )

        self.screen_w = self.width * cell_size_px
        self.screen_h = self.height * cell_size_px
        if not headless:
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("GridWorld_c2 (Continuous Action)")
        else:
            self.screen = None
        self.clock = pygame.time.Clock()

        self.wall_rects = []
        self.trap_rects = []
        self.goal_rects = []
        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    c * cell_size_px,
                    r * cell_size_px,
                    cell_size_px,
                    cell_size_px,
                )
                cell_type = self.grid[r, c]
                if cell_type == 1:
                    self.wall_rects.append(rect)
                elif cell_type == 2:
                    self.trap_rects.append(rect)
                elif cell_type == 3:
                    self.goal_rects.append(rect)

        self.agent_pos = None
        self.done = False

    def reset(
        self, start_pos: Optional[Sequence[Union[int, float]]] = None
    ) -> np.ndarray:
        if start_pos is None:
            r, c = self.start_cell
        else:
            r, c = start_pos
        self.agent_pos = np.array([r + 0.5, c + 0.5], dtype=float) * self.cell_size_m
        self.done = False
        return self._get_obs()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")

        delta = np.clip(np.asarray(action, dtype=float), -1.0, 1.0) * self.step_size_m
        old_pos = self.agent_pos.copy()

        lo, hi = 0.0, 1.0
        for _ in range(8):
            mid = (lo + hi) / 2
            test_pos = old_pos + delta * mid
            px = test_pos[1] / self.cell_size_m * self.cell_size_px
            py = test_pos[0] / self.cell_size_m * self.cell_size_px
            rect = pygame.Rect(
                px - self.agent_radius,
                py - self.agent_radius,
                self.agent_radius * 2,
                self.agent_radius * 2,
            )

            wall_hit = any(rect.colliderect(wall) for wall in self.wall_rects)
            out_of_bounds = (
                rect.left < 0
                or rect.top < 0
                or rect.right > self.screen_w
                or rect.bottom > self.screen_h
            )

            if wall_hit or out_of_bounds:
                hi = mid
            else:
                lo = mid

        self.agent_pos = old_pos + delta * lo

        px = self.agent_pos[1] / self.cell_size_m * self.cell_size_px
        py = self.agent_pos[0] / self.cell_size_m * self.cell_size_px
        agent_rect = pygame.Rect(
            px - self.agent_radius,
            py - self.agent_radius,
            self.agent_radius * 2,
            self.agent_radius * 2,
        )

        if any(agent_rect.colliderect(trap) for trap in self.trap_rects):
            self.done = True
            return self._get_obs(), -100.0, True, {}
        if any(agent_rect.colliderect(goal) for goal in self.goal_rects):
            self.done = True
            return self._get_obs(), 100.0, True, {}
        return self._get_obs(), -1.0, False, {}

    def _get_obs(self) -> np.ndarray:
        return self.agent_pos.copy()

    def render(self, tick: int = 30) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(
                    c * self.cell_size_px,
                    r * self.cell_size_px,
                    self.cell_size_px,
                    self.cell_size_px,
                )
                cell_type = self.grid[r, c]
                if cell_type == 1:
                    color = (50, 50, 50)
                elif cell_type == 2:
                    color = (200, 0, 0)
                elif cell_type == 3:
                    color = (0, 200, 0)
                else:
                    color = (220, 220, 220)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 1)

        px = self.agent_pos[1] / self.cell_size_m * self.cell_size_px
        py = self.agent_pos[0] / self.cell_size_m * self.cell_size_px
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),
            (int(px), int(py)),
            self.agent_radius,
        )
        pygame.display.flip()
        self.clock.tick(tick)

    def close(self) -> None:
        pygame.quit()
