"""Tests for the Ascent env, agents, and training."""

from __future__ import annotations

import numpy as np
import pytest

from ascent import MountainCar, QLearningAgent, SarsaAgent, evaluate, make_agent, train


# ------------------------------- env --------------------------------------- #
def test_reset_is_in_start_band():
    env = MountainCar()
    pos, vel = env.reset(seed=0)
    assert -0.6 <= pos <= -0.4
    assert vel == 0.0


def test_reset_is_reproducible():
    a = MountainCar().reset(seed=42)
    b = MountainCar().reset(seed=42)
    assert a == b


def test_physics_clamps_speed_and_bounds():
    env = MountainCar()
    env.reset(seed=1)
    for _ in range(50):
        _, _, term, trunc = env.step(2)  # always push right
        assert -env.cfg.max_speed <= env.velocity <= env.cfg.max_speed
        assert env.cfg.min_position <= env.position <= env.cfg.max_position
        if term or trunc:
            break


def test_reward_is_minus_one_until_goal():
    env = MountainCar()
    env.reset(seed=3)
    _, r, term, _ = env.step(1)
    assert r == -1.0 and term is False


def test_truncates_at_max_steps():
    env = MountainCar()
    env.reset(seed=5)
    term = trunc = False
    for _ in range(env.cfg.max_steps):
        _, _, term, trunc = env.step(1)  # idle: never reaches goal
    assert trunc is True and term is False


def test_invalid_action_raises():
    env = MountainCar()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(5)


# ------------------------------ agents ------------------------------------- #
def test_discretize_in_range():
    agent = QLearningAgent(n_bins=40)
    pi, vi = agent.discretize((-1.2, -0.07))
    assert (pi, vi) == (0, 0)
    pi, vi = agent.discretize((0.6, 0.07))
    assert pi == 39 and vi == 39


def test_greedy_action_follows_Q():
    agent = QLearningAgent(n_bins=40)
    pi, vi = agent.discretize((0.0, 0.0))
    agent.Q[pi, vi] = [1.0, 0.0, 5.0]
    assert agent.select_action((0.0, 0.0), greedy=True) == 2


def test_epsilon_decays_to_floor():
    agent = QLearningAgent(eps_start=1.0, eps_end=0.1, eps_decay=0.5)
    for _ in range(100):
        agent.decay_epsilon()
    assert agent.epsilon == 0.1


def test_grids_have_right_shape():
    agent = QLearningAgent(n_bins=40)
    assert agent.policy_grid().shape == (40, 40)
    assert agent.value_grid().shape == (40, 40)
    assert agent.visit_grid().shape == (40, 40)


def test_make_agent():
    assert isinstance(make_agent("qlearning"), QLearningAgent)
    assert isinstance(make_agent("sarsa"), SarsaAgent)
    with pytest.raises(ValueError):
        make_agent("nope")


# ----------------------------- learning ------------------------------------ #
def test_qlearning_actually_learns_to_climb():
    """After training, the greedy policy should reach the goal most of the time
    — far better than an untrained agent (which times out every episode)."""
    agent = QLearningAgent(seed=0)
    before = evaluate(agent, n_episodes=20)
    assert before["success_rate"] == 0.0  # untrained never makes it

    train(agent, n_episodes=4000, eval_every=4000, n_eval=10)
    after = evaluate(agent, n_episodes=50)
    assert after["success_rate"] >= 0.5
    assert after["mean"] > before["mean"]  # less negative = fewer steps


def test_training_is_reproducible():
    h1 = train(QLearningAgent(seed=7), n_episodes=300, eval_every=300, n_eval=5)
    h2 = train(QLearningAgent(seed=7), n_episodes=300, eval_every=300, n_eval=5)
    assert h1["rewards"] == h2["rewards"]
