"""Ascent — tabular reinforcement learning on MountainCar.

A small, dependency-light library: a pure-Python MountainCar environment,
Q-Learning and SARSA agents over a discretised state grid, and training /
evaluation loops with the standard mean / std / success-rate metrics.
"""

from .agents.tabular import QLearningAgent, SarsaAgent
from .envs.mountain_car import MountainCar, MountainCarConfig
from .training import evaluate, make_agent, train

__all__ = [
    "MountainCar",
    "MountainCarConfig",
    "QLearningAgent",
    "SarsaAgent",
    "train",
    "evaluate",
    "make_agent",
]
__version__ = "0.1.0"
