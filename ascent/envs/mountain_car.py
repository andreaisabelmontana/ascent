"""Classic MountainCar environment — pure Python, no Gymnasium dependency.

An underpowered car sits in a valley between two hills; the engine alone can't
climb the right hill, so the agent must learn to rock back and forth to build
momentum. Dynamics match the canonical MountainCar-v0 (Moore, 1990).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Action ids: 0 = push left, 1 = no push, 2 = push right.
ACTIONS = (0, 1, 2)


@dataclass(frozen=True)
class MountainCarConfig:
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.5
    force: float = 0.001
    gravity: float = 0.0025
    max_steps: int = 200


class MountainCar:
    """Minimal MountainCar with a Gym-like step/reset interface."""

    def __init__(self, config: MountainCarConfig | None = None):
        self.cfg = config or MountainCarConfig()
        self.position = 0.0
        self.velocity = 0.0
        self.steps = 0
        self._rng = random.Random()

    def reset(self, seed: int | None = None) -> tuple[float, float]:
        if seed is not None:
            self._rng.seed(seed)
        # Standard start: random position in [-0.6, -0.4], zero velocity.
        self.position = self._rng.uniform(-0.6, -0.4)
        self.velocity = 0.0
        self.steps = 0
        return self.state

    @property
    def state(self) -> tuple[float, float]:
        return (self.position, self.velocity)

    def step(self, action: int) -> tuple[tuple[float, float], float, bool, bool]:
        """Advance one timestep. Returns (state, reward, terminated, truncated)."""
        if action not in ACTIONS:
            raise ValueError(f"action must be 0, 1 or 2 — got {action}")
        c = self.cfg

        self.velocity += (action - 1) * c.force + math.cos(3 * self.position) * (-c.gravity)
        self.velocity = max(-c.max_speed, min(c.max_speed, self.velocity))
        self.position += self.velocity
        self.position = max(c.min_position, min(c.max_position, self.position))
        # Hitting the left wall kills momentum.
        if self.position <= c.min_position and self.velocity < 0:
            self.velocity = 0.0

        self.steps += 1
        terminated = self.position >= c.goal_position
        truncated = self.steps >= c.max_steps
        reward = 0.0 if terminated else -1.0
        return self.state, reward, terminated, truncated

    @staticmethod
    def height(position: float) -> float:
        """Terrain height at a position (for rendering the hill)."""
        return math.sin(3 * position) * 0.45 + 0.55
