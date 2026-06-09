"""Tabular agents with uniform state-space discretization.

Q-Learning (off-policy) and SARSA (on-policy) over a binned (position, velocity)
grid. Mirrors the structure of the original study: a 40x40x3 table, epsilon-
greedy behaviour with exponential decay, and grid extractors for the value
surface, greedy policy, and visit frequency used by the visualisations.
"""

from __future__ import annotations

import numpy as np

POS_RANGE = (-1.2, 0.6)
VEL_RANGE = (-0.07, 0.07)


class QLearningAgent:
    def __init__(self, n_bins=40, n_actions=3, alpha=0.2, gamma=0.99,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.999, seed=0):
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.pos_bins = np.linspace(*POS_RANGE, n_bins + 1)
        self.vel_bins = np.linspace(*VEL_RANGE, n_bins + 1)
        self.Q = np.zeros((n_bins, n_bins, n_actions))
        self.N = np.zeros((n_bins, n_bins, n_actions), dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def discretize(self, state):
        pos, vel = state
        pi = int(np.clip(np.digitize(pos, self.pos_bins) - 1, 0, self.n_bins - 1))
        vi = int(np.clip(np.digitize(vel, self.vel_bins) - 1, 0, self.n_bins - 1))
        return pi, vi

    def select_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        pi, vi = self.discretize(state)
        return int(np.argmax(self.Q[pi, vi]))

    def update(self, state, action, reward, next_state, done):
        pi, vi = self.discretize(state)
        npi, nvi = self.discretize(next_state)
        self.N[pi, vi, action] += 1
        target = reward if done else reward + self.gamma * np.max(self.Q[npi, nvi])
        self.Q[pi, vi, action] += self.alpha * (target - self.Q[pi, vi, action])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # --- grids for visualisation -----------------------------------------
    def policy_grid(self):
        return np.argmax(self.Q, axis=2)

    def value_grid(self):
        return np.max(self.Q, axis=2)

    def visit_grid(self):
        return np.sum(self.N, axis=2)


class SarsaAgent(QLearningAgent):
    """On-policy control: the TD target uses the action actually taken next."""

    def update(self, state, action, reward, next_state, next_action, done):
        pi, vi = self.discretize(state)
        npi, nvi = self.discretize(next_state)
        self.N[pi, vi, action] += 1
        target = reward if done else reward + self.gamma * self.Q[npi, nvi, next_action]
        self.Q[pi, vi, action] += self.alpha * (target - self.Q[pi, vi, action])
