"""Training and evaluation loops + summary metrics."""

from __future__ import annotations

import numpy as np

from .agents.tabular import QLearningAgent, SarsaAgent
from .envs.mountain_car import MountainCar, MountainCarConfig


def _reward(base: float, velocity: float, shaping: bool) -> float:
    """Optional reward shaping: add a small bonus for speed to encourage the
    momentum-building behaviour the task actually needs."""
    return base + (10.0 * abs(velocity) if shaping else 0.0)


def train(agent, env: MountainCar | None = None, n_episodes=4000,
          shaping=False, eval_every=500, n_eval=50):
    """Train a Q-Learning or SARSA agent; returns a history dict."""
    env = env or MountainCar()
    is_sarsa = isinstance(agent, SarsaAgent)
    rewards: list[float] = []
    eval_eps: list[int] = []
    eval_means: list[float] = []
    eval_stds: list[float] = []

    for ep in range(n_episodes):
        state = env.reset(seed=ep)
        action = agent.select_action(state)
        total = 0.0
        while True:
            nxt, base_r, term, trunc = env.step(action)
            r = _reward(base_r, nxt[1], shaping)
            if is_sarsa:
                next_action = agent.select_action(nxt)
                agent.update(state, action, r, nxt, next_action, term)
                action = next_action
            else:
                agent.update(state, action, r, nxt, term)
                action = agent.select_action(nxt)
            state = nxt
            total += base_r  # report the *true* objective, not the shaped reward
            if term or trunc:
                break
        agent.decay_epsilon()
        rewards.append(total)

        if (ep + 1) % eval_every == 0:
            res = evaluate(agent, env, n_episodes=n_eval)
            eval_eps.append(ep + 1)
            eval_means.append(res["mean"])
            eval_stds.append(res["std"])

    return {
        "rewards": rewards,
        "eval_episodes": eval_eps,
        "eval_means": eval_means,
        "eval_stds": eval_stds,
    }


def evaluate(agent, env: MountainCar | None = None, n_episodes=100) -> dict:
    """Greedy evaluation; returns mean/std reward, success rate, mean steps."""
    env = env or MountainCar()
    rewards, steps, successes = [], [], 0
    for ep in range(n_episodes):
        state = env.reset(seed=10_000 + ep)
        total = 0.0
        reached = False
        while True:
            action = agent.select_action(state, greedy=True)
            state, r, term, trunc = env.step(action)
            total += r
            if term:
                reached = True
            if term or trunc:
                break
        rewards.append(total)
        steps.append(env.steps)
        successes += int(reached)
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "success_rate": successes / n_episodes,
        "mean_steps": float(np.mean(steps)),
        "rewards": rewards,
    }


def make_agent(kind: str, **kwargs):
    if kind == "qlearning":
        return QLearningAgent(**kwargs)
    if kind == "sarsa":
        return SarsaAgent(**kwargs)
    raise ValueError(f"unknown agent kind: {kind!r}")
