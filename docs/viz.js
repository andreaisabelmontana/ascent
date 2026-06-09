/* Ascent — canvas renderers: heatmaps, learning curve, hill animation,
   phase portrait. Grids are [vi][pi]; we draw position on x, velocity on y
   (high velocity at the top). */

"use strict";

const Viz = (() => {
  const N = 40;

  function lerp(a, b, t) { return a + (b - a) * t; }
  function hex(c) {
    return "#" + c.map((x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, "0")).join("");
  }
  // Dark-purple -> magenta -> orange -> pale-yellow (magma-ish).
  const STOPS = [[13, 8, 38], [84, 15, 110], [187, 55, 84], [249, 142, 9], [252, 253, 191]];
  function magma(t) {
    t = Math.max(0, Math.min(1, t));
    const x = t * (STOPS.length - 1);
    const i = Math.min(STOPS.length - 2, Math.floor(x));
    const f = x - i;
    return hex([0, 1, 2].map((k) => lerp(STOPS[i][k], STOPS[i + 1][k], f)));
  }

  function cellSize(canvas) { return canvas.width / N; }

  function heatmap(canvas, grid, mode) {
    const ctx = canvas.getContext("2d");
    const cw = canvas.width / N, ch = canvas.height / N;
    let min = Infinity, max = -Infinity;
    for (const row of grid) for (const v of row) { if (v < min) min = v; if (v > max) max = v; }
    const span = max - min || 1;
    const maxVisit = mode === "visit" ? max : 0;

    for (let vi = 0; vi < N; vi++) {
      for (let pi = 0; pi < N; pi++) {
        const v = grid[vi][pi];
        let fill;
        if (mode === "policy") fill = ["#ef4444", "#cbd5e1", "#22c55e"][v];
        else if (mode === "visit") fill = v === 0 ? "#0b0820" : magma(Math.log1p(v) / Math.log1p(maxVisit));
        else fill = magma((v - min) / span);
        ctx.fillStyle = fill;
        ctx.fillRect(pi * cw, (N - 1 - vi) * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
  }

  function lineChart(canvas, values, { smooth = true } = {}) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height, pad = 24;
    ctx.clearRect(0, 0, W, H);
    if (!values.length) return;
    let lo = Math.min(...values), hi = Math.max(...values);
    if (hi === lo) { hi += 1; lo -= 1; }
    // axes
    ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - 6, H - pad); ctx.stroke();
    ctx.fillStyle = "#94a3b8"; ctx.font = "10px sans-serif";
    ctx.fillText(Math.round(hi), 2, pad + 4);
    ctx.fillText(Math.round(lo), 2, H - pad);

    const sm = [];
    if (smooth) {
      const k = Math.max(1, Math.floor(values.length / 80));
      for (let i = 0; i < values.length; i += k) {
        const slice = values.slice(i, i + k);
        sm.push(slice.reduce((a, b) => a + b, 0) / slice.length);
      }
    }
    const data = smooth ? sm : values;
    ctx.strokeStyle = "#7c3aed"; ctx.lineWidth = 1.8; ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad + (i / Math.max(1, data.length - 1)) * (W - pad - 6);
      const y = pad + (1 - (v - lo) / (hi - lo)) * (H - 2 * pad);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }

  function drawHill(canvas, carPos) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const X = (p) => ((p - (-1.2)) / 1.8) * W;
    const Y = (p) => H - (MountainCar.height(p) / 1.1) * (H - 20) - 8;

    // terrain
    ctx.beginPath(); ctx.moveTo(0, H);
    for (let p = -1.2; p <= 0.6; p += 0.02) ctx.lineTo(X(p), Y(p));
    ctx.lineTo(W, H); ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, "#ede9fe"); grad.addColorStop(1, "#c4b5fd");
    ctx.fillStyle = grad; ctx.fill();
    ctx.strokeStyle = "#7c3aed"; ctx.lineWidth = 2; ctx.stroke();

    // goal flag at 0.5
    const gx = X(0.5), gy = Y(0.5);
    ctx.strokeStyle = "#334155"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(gx, gy - 34); ctx.stroke();
    ctx.fillStyle = "#22c55e"; ctx.fillRect(gx, gy - 34, 18, 12);

    // car
    const cx = X(carPos), cy = Y(carPos);
    ctx.fillStyle = "#1f2937"; ctx.beginPath(); ctx.arc(cx, cy - 6, 7, 0, 2 * Math.PI); ctx.fill();
  }

  function phasePortrait(canvas, traj) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height, pad = 22;
    ctx.clearRect(0, 0, W, H);
    const X = (p) => pad + ((p - (-1.2)) / 1.8) * (W - pad - 6);
    const Y = (v) => pad + (1 - (v + 0.07) / 0.14) * (H - 2 * pad);
    ctx.strokeStyle = "#e5e7eb"; ctx.beginPath();
    ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - 6, H - pad); ctx.stroke();
    ctx.fillStyle = "#94a3b8"; ctx.font = "10px sans-serif";
    ctx.fillText("vel", 2, pad - 6); ctx.fillText("pos", W - 26, H - 6);
    // zero-velocity line
    ctx.strokeStyle = "#f1f5f9"; ctx.beginPath(); ctx.moveTo(pad, Y(0)); ctx.lineTo(W - 6, Y(0)); ctx.stroke();
    if (!traj || !traj.length) return;
    ctx.strokeStyle = "#7c3aed"; ctx.lineWidth = 1.6; ctx.beginPath();
    traj.forEach((s, i) => (i ? ctx.lineTo(X(s[0]), Y(s[1])) : ctx.moveTo(X(s[0]), Y(s[1]))));
    ctx.stroke();
    // goal line
    ctx.strokeStyle = "#22c55e"; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(X(0.5), pad); ctx.lineTo(X(0.5), H - pad); ctx.stroke();
    ctx.setLineDash([]);
  }

  return { heatmap, lineChart, drawHill, phasePortrait, cellSize };
})();
