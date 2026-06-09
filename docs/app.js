/* Ascent — playground UI. Trains a tabular agent live in the browser and
   renders the value surface, greedy policy, visit map, learning curve, a hill
   animation and the phase portrait. */

"use strict";

const $ = (id) => document.getElementById(id);

const env = new MountainCar();
let agent = null;
let history = [];
let target = 3000;
let epDone = 0;
let training = false;
let animTimer = null;

const els = {
  algo: $("algo"), alpha: $("alpha"), gamma: $("gamma"), decay: $("decay"),
  shaping: $("shaping"), episodes: $("episodes"),
  alphaV: $("alphaV"), gammaV: $("gammaV"), decayV: $("decayV"), episodesV: $("episodesV"),
  train: $("train"), reset: $("reset"), rollout: $("rollout"), evalBtn: $("evalBtn"),
  epInfo: $("epInfo"), epsInfo: $("epsInfo"), metrics: $("metrics"),
  curve: $("curve"), value: $("value"), policy: $("policy"), visit: $("visit"),
  hill: $("hill"), phase: $("phase"),
};

function newAgent() {
  agent = new TabularAgent({
    kind: els.algo.value,
    alpha: +els.alpha.value,
    gamma: +els.gamma.value,
    epsDecay: +els.decay.value,
    seed: 0,
  });
  history = [];
  epDone = 0;
}

function syncLabels() {
  els.alphaV.textContent = (+els.alpha.value).toFixed(2);
  els.gammaV.textContent = (+els.gamma.value).toFixed(3);
  els.decayV.textContent = (+els.decay.value).toFixed(4);
  els.episodesV.textContent = els.episodes.value;
}

function redrawGrids() {
  Viz.heatmap(els.value, agent.valueGrid(), "value");
  Viz.heatmap(els.policy, agent.policyGrid(), "policy");
  Viz.heatmap(els.visit, agent.visitGrid(), "visit");
}

function readouts() {
  els.epInfo.textContent = `${epDone} / ${target}`;
  els.epsInfo.textContent = agent.epsilon.toFixed(3);
}

function trainLoop() {
  if (!training) return;
  const chunk = 40;
  for (let i = 0; i < chunk && epDone < target; i++) {
    history.push(runEpisode(env, agent, epDone, els.shaping.checked));
    epDone++;
  }
  readouts();
  Viz.lineChart(els.curve, history);
  if (epDone % 200 === 0 || epDone >= target) redrawGrids();
  if (epDone >= target) { training = false; els.train.textContent = "▶ Train"; onTrainDone(); return; }
  requestAnimationFrame(trainLoop);
}

function onTrainDone() {
  runRollout();
  runEval();
}

function runRollout() {
  if (animTimer) clearInterval(animTimer);
  const { traj } = greedyRollout(env, agent, 777);
  Viz.phasePortrait(els.phase, traj);
  let i = 0;
  animTimer = setInterval(() => {
    if (i >= traj.length) { clearInterval(animTimer); animTimer = null; return; }
    Viz.drawHill(els.hill, traj[i][0]);
    i++;
  }, 16);
}

function runEval() {
  const m = evaluate(env, agent, 100);
  els.metrics.innerHTML = [
    ["Success rate", (m.successRate * 100).toFixed(0) + "%", m.successRate >= 0.5 ? "#16a34a" : "#dc2626"],
    ["Mean reward", m.mean.toFixed(1), "#7c3aed"],
    ["Std", m.std.toFixed(1), "#64748b"],
    ["Mean steps", m.meanSteps.toFixed(0), "#0ea5e9"],
  ].map(([k, v, c]) => `<div class="metric"><div class="m-v" style="color:${c}">${v}</div><div class="m-k">${k}</div></div>`).join("");
}

function startTraining() {
  if (training) { training = false; els.train.textContent = "▶ Train"; return; }
  newAgent();
  target = +els.episodes.value;
  training = true;
  els.train.textContent = "⏸ Stop";
  els.metrics.innerHTML = "";
  requestAnimationFrame(trainLoop);
}

function reset() {
  training = false;
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
  els.train.textContent = "▶ Train";
  newAgent();
  readouts();
  redrawGrids();
  Viz.lineChart(els.curve, []);
  Viz.drawHill(els.hill, -0.5);
  Viz.phasePortrait(els.phase, []);
  els.metrics.innerHTML = '<p class="hint">Train an agent, then watch it roll.</p>';
}

[els.alpha, els.gamma, els.decay, els.episodes].forEach((s) => s.addEventListener("input", syncLabels));
els.train.addEventListener("click", startTraining);
els.reset.addEventListener("click", reset);
els.rollout.addEventListener("click", () => agent && runRollout());
els.evalBtn.addEventListener("click", () => agent && runEval());

syncLabels();
reset();
