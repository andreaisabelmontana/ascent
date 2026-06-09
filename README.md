# ⛰️ Ascent

Tabular **reinforcement learning** on the classic **MountainCar** problem — as a
tested Python library *and* a live, in-browser playground.

**▶️ Live playground:** https://andreaisabelmontana.github.io/ascent/

An underpowered car sits in a valley; the engine alone can't climb the right
hill, so the agent has to learn to **build momentum** by rocking back and forth.
Watch a Q-Learning or SARSA agent figure that out, live, and see its value
surface, greedy policy, and visit map emerge.

## Two parts

| Part | What it is | Where |
|------|------------|-------|
| **Library** | A pure-Python MountainCar env, Q-Learning & SARSA agents over a 40×40 discretised grid, and training/evaluation loops with mean / std / success-rate / mean-steps metrics. | `ascent/` |
| **Playground** | The same env + agents reimplemented in vanilla JS, trained **live in the browser**, with value/policy/visit heatmaps, a learning curve, a hill animation, and the phase portrait. | `docs/` (GitHub Pages) |

The browser agent mirrors the Python update rules, so both learn the same way.

## What you can see

- **Value surface** `V(s)=maxₐ Q(s,a)` over (position, velocity).
- **Greedy policy** map — where the agent pushes left / idles / pushes right.
- **Visit frequency** — which states the agent actually explored.
- **Phase portrait** — the back-and-forth momentum loops in (pos, vel) space.
- **Greedy evaluation** — success rate, mean reward, std, mean steps over 100 episodes.
- Knobs for **α, γ, exploration decay, episodes**, a **Q-Learning vs SARSA** switch,
  and an optional **reward-shaping** (speed-bonus) toggle.

> Tabular RL is genuinely unreliable run-to-run — more episodes don't always
> mean a better greedy policy. That high variance is a real lesson, not a bug;
> the evaluation panel reports std and success rate precisely so you can see it.

## Use the library

```bash
pip install -e ".[dev]"
```

```python
from ascent import QLearningAgent, train, evaluate

agent = QLearningAgent(alpha=0.2, gamma=0.99)
train(agent, n_episodes=4000)
print(evaluate(agent, n_episodes=100))
# {'mean': -1xx, 'std': ..., 'success_rate': 0.9x, 'mean_steps': 1xx, ...}
```

## Tests

```bash
pytest
```

Covers the environment physics, discretisation, ε-decay, the grid extractors,
training reproducibility, and an end-to-end check that Q-Learning actually
learns to reach the goal (0% → >50% success after training).

## Layout

```
ascent/
  envs/mountain_car.py   pure-Python MountainCar dynamics
  agents/tabular.py      Q-Learning + SARSA over a discretised grid
  training.py            train / evaluate loops + metrics
docs/                    live playground (GitHub Pages): rl.js, viz.js, app.js
tests/                   pytest suite
```

## License

MIT — see [LICENSE](LICENSE).
