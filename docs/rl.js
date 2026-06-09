/* Ascent — MountainCar environment + tabular agents, in the browser.
   Mirrors ascent/envs/mountain_car.py and ascent/agents/tabular.py so the
   in-browser run matches the Python library. */

"use strict";

const ENV = {
  minPos: -1.2, maxPos: 0.6, maxSpeed: 0.07, goal: 0.5,
  force: 0.001, gravity: 0.0025, maxSteps: 200,
};

function mulberry32(seed) {
  // Small deterministic PRNG so runs are reproducible.
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class MountainCar {
  constructor() {
    this.rng = mulberry32(1);
    this.reset(0);
  }
  reset(seed) {
    if (seed !== undefined) this.rng = mulberry32(seed + 1);
    this.position = -0.6 + this.rng() * 0.2; // [-0.6, -0.4]
    this.velocity = 0;
    this.steps = 0;
    return [this.position, this.velocity];
  }
  step(action) {
    this.velocity += (action - 1) * ENV.force + Math.cos(3 * this.position) * -ENV.gravity;
    this.velocity = Math.max(-ENV.maxSpeed, Math.min(ENV.maxSpeed, this.velocity));
    this.position += this.velocity;
    this.position = Math.max(ENV.minPos, Math.min(ENV.maxPos, this.position));
    if (this.position <= ENV.minPos && this.velocity < 0) this.velocity = 0;
    this.steps++;
    const terminated = this.position >= ENV.goal;
    const truncated = this.steps >= ENV.maxSteps;
    return [[this.position, this.velocity], terminated ? 0 : -1, terminated, truncated];
  }
}
MountainCar.height = (p) => Math.sin(3 * p) * 0.45 + 0.55;

const NB = 40, NA = 3;
const POS_W = (ENV.maxPos - ENV.minPos) / NB;
const VEL_W = (ENV.maxSpeed * 2) / NB;

class TabularAgent {
  constructor(opts = {}) {
    this.kind = opts.kind || "qlearning";
    this.alpha = opts.alpha ?? 0.2;
    this.gamma = opts.gamma ?? 0.99;
    this.epsilon = 1.0;
    this.epsEnd = opts.epsEnd ?? 0.01;
    this.epsDecay = opts.epsDecay ?? 0.999;
    this.Q = new Float64Array(NB * NB * NA);
    this.N = new Int32Array(NB * NB * NA);
    this.rng = mulberry32(opts.seed ?? 0);
  }
  _di(pos, vel) {
    let pi = Math.floor((pos - ENV.minPos) / POS_W);
    let vi = Math.floor((vel + ENV.maxSpeed) / VEL_W);
    pi = Math.max(0, Math.min(NB - 1, pi));
    vi = Math.max(0, Math.min(NB - 1, vi));
    return pi * NB * NA + vi * NA;
  }
  _argmax(base) {
    let best = 0, bv = this.Q[base];
    for (let a = 1; a < NA; a++) if (this.Q[base + a] > bv) { bv = this.Q[base + a]; best = a; }
    return best;
  }
  _max(base) {
    let m = this.Q[base];
    for (let a = 1; a < NA; a++) if (this.Q[base + a] > m) m = this.Q[base + a];
    return m;
  }
  selectAction(state, greedy = false) {
    if (!greedy && this.rng() < this.epsilon) return Math.floor(this.rng() * NA);
    return this._argmax(this._di(state[0], state[1]));
  }
  updateQ(s, a, r, ns, done) {
    const b = this._di(s[0], s[1]);
    this.N[b + a]++;
    const target = done ? r : r + this.gamma * this._max(this._di(ns[0], ns[1]));
    this.Q[b + a] += this.alpha * (target - this.Q[b + a]);
  }
  updateSarsa(s, a, r, ns, na, done) {
    const b = this._di(s[0], s[1]);
    this.N[b + a]++;
    const target = done ? r : r + this.gamma * this.Q[this._di(ns[0], ns[1]) + na];
    this.Q[b + a] += this.alpha * (target - this.Q[b + a]);
  }
  decayEpsilon() { this.epsilon = Math.max(this.epsEnd, this.epsilon * this.epsDecay); }

  // Grids for visualisation: [vi][pi] so rows = velocity, cols = position.
  valueGrid() {
    const g = [];
    for (let vi = 0; vi < NB; vi++) { const row = []; for (let pi = 0; pi < NB; pi++) row.push(this._max((pi * NB + vi) * NA)); g.push(row); }
    return g;
  }
  policyGrid() {
    const g = [];
    for (let vi = 0; vi < NB; vi++) { const row = []; for (let pi = 0; pi < NB; pi++) row.push(this._argmax((pi * NB + vi) * NA)); g.push(row); }
    return g;
  }
  visitGrid() {
    const g = [];
    for (let vi = 0; vi < NB; vi++) {
      const row = [];
      for (let pi = 0; pi < NB; pi++) { const b = (pi * NB + vi) * NA; row.push(this.N[b] + this.N[b + 1] + this.N[b + 2]); }
      g.push(row);
    }
    return g;
  }
}

/* One training episode; returns the true (unshaped) total reward. */
function runEpisode(env, agent, seed, shaping) {
  let s = env.reset(seed);
  let a = agent.selectAction(s);
  let total = 0;
  while (true) {
    const [ns, baseR, term, trunc] = env.step(a);
    const r = baseR + (shaping ? 10 * Math.abs(ns[1]) : 0);
    if (agent.kind === "sarsa") {
      const na = agent.selectAction(ns);
      agent.updateSarsa(s, a, r, ns, na, term);
      a = na;
    } else {
      agent.updateQ(s, a, r, ns, term);
      a = agent.selectAction(ns);
    }
    s = ns; total += baseR;
    if (term || trunc) break;
  }
  agent.decayEpsilon();
  return total;
}

/* Greedy rollout; returns the trajectory + outcome for animation/metrics. */
function greedyRollout(env, agent, seed) {
  const traj = [];
  let s = env.reset(seed);
  let reached = false;
  while (true) {
    traj.push([s[0], s[1]]);
    const a = agent.selectAction(s, true);
    const [ns, , term, trunc] = env.step(a);
    s = ns;
    if (term) { reached = true; traj.push([s[0], s[1]]); break; }
    if (trunc) break;
  }
  return { traj, reached, steps: env.steps };
}

function evaluate(env, agent, n = 100) {
  let rewards = [], steps = [], succ = 0;
  for (let i = 0; i < n; i++) {
    const r = greedyRollout(env, agent, 10000 + i);
    rewards.push(r.reached ? -(r.steps - 1) : -200);
    steps.push(r.steps);
    succ += r.reached ? 1 : 0;
  }
  const mean = rewards.reduce((a, b) => a + b, 0) / n;
  const variance = rewards.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  return { mean, std: Math.sqrt(variance), successRate: succ / n,
           meanSteps: steps.reduce((a, b) => a + b, 0) / n };
}
