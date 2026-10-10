import { Sim } from './sim.js';

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const $ = id => document.getElementById(id);
const sim = new Sim();

let dpr = 1;
function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = innerWidth * dpr;
  canvas.height = innerHeight * dpr;
  canvas.style.width = innerWidth + 'px';
  canvas.style.height = innerHeight + 'px';
  sim.bounds.w = innerWidth;
  sim.bounds.h = innerHeight;
}
addEventListener('resize', resize);
resize();

function build() {
  const spacing = Math.max(12, Math.min(18, innerWidth / 36));
  const nx = Math.min(26, Math.floor((innerWidth - 80) / spacing));
  const ny = 8;
  const ox = Math.max(20, (innerWidth - nx * spacing) / 2);
  sim.buildSlab(nx, ny, spacing, ox, 140, $('pin').checked);
}

function applyControls() {
  sim.compliance = 1e-7 * Math.pow(10, +$('soft').value / 100 * 5); // 1e-7 .. 1e-2
  sim.areaEnabled = $('area').checked;
  sim.areaCompliance = 1e-7 * Math.pow(10, +$('softA').value / 100 * 5);
  sim.gravity = +$('grav').value;
  sim.tearEnabled = $('tear').checked;
  sim.tearStrain = +$('tearAt').value / 100;
}
['soft', 'softA', 'area', 'grav', 'tear', 'tearAt'].forEach(id => $(id).addEventListener('input', applyControls));
$('pin').addEventListener('change', build);
$('reset').addEventListener('click', build);
applyControls();
build();

// --- перетаскивание мышью/пальцем: вершина становится кинематической ---
let drag = -1, dragW = 1;
function pointer(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
canvas.addEventListener('pointerdown', e => {
  const [mx, my] = pointer(e);
  const k = sim.pick(mx, my, 40);
  if (k >= 0) { drag = k; dragW = sim.w[k]; sim.w[k] = 0; canvas.setPointerCapture(e.pointerId); }
});
canvas.addEventListener('pointermove', e => {
  if (drag < 0) return;
  const [mx, my] = pointer(e);
  sim.x[drag * 2] = mx; sim.x[drag * 2 + 1] = my;
});
function release() { if (drag >= 0) { sim.w[drag] = dragW; drag = -1; } }
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);

// --- рендер ---
function strainColor(s) {
  const t = Math.min(1, s / 0.3);
  const hue = 210 - t * 210; // синий → красный
  return `hsl(${hue.toFixed(0)}, 75%, ${(38 + t * 12).toFixed(0)}%)`;
}

function draw() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  const { x, tris, triEdges, alive, strain } = sim;
  const T = tris.length / 3;

  for (let t = 0; t < T; t++) {
    const e0 = triEdges[t * 3], e1 = triEdges[t * 3 + 1], e2 = triEdges[t * 3 + 2];
    if (!alive[e0] || !alive[e1] || !alive[e2]) continue; // наивно: треугольник с порванным ребром не рисуем
    const a = tris[t * 3], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
    ctx.beginPath();
    ctx.moveTo(x[a * 2], x[a * 2 + 1]);
    ctx.lineTo(x[b * 2], x[b * 2 + 1]);
    ctx.lineTo(x[c * 2], x[c * 2 + 1]);
    ctx.closePath();
    ctx.fillStyle = strainColor(Math.max(strain[e0], strain[e1], strain[e2]));
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 0.6;
    ctx.stroke();
  }

  // пол
  ctx.strokeStyle = '#33465f';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0, innerHeight - 1); ctx.lineTo(innerWidth, innerHeight - 1); ctx.stroke();

  // закреплённые вершины
  ctx.fillStyle = '#ffd24f';
  for (let i = 0; i < sim.n; i++) if (sim.w[i] === 0 && i !== drag) {
    ctx.beginPath(); ctx.arc(x[i * 2], x[i * 2 + 1], 2.5, 0, Math.PI * 2); ctx.fill();
  }
}

// --- главный цикл: фиксированный шаг 60 Гц ---
let last = performance.now(), acc = 0;
const DT = 1 / 60;
function frame(now) {
  acc += Math.min(0.1, (now - last) / 1000);
  last = now;
  while (acc >= DT) { sim.step(DT); acc -= DT; }
  draw();
  $('info').textContent = `частиц: ${sim.n} · рёбер: ${sim.rest.length} · вывернуто: ${sim.invertedCount()} · E: ${sim.energy().toFixed(0)}`;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
