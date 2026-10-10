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
  sim.femEnabled = $('fem').checked;
  sim.cutEnabled = $('cut').checked;
  sim.collideEnabled = $('collide').checked;
  sim.young = 1e4 * Math.pow(10, +$('young').value / 100 * 4); // 1e4 .. 1e8
  sim.nu = +$('nu').value / 100;
  sim.areaEnabled = $('area').checked;
  sim.areaCompliance = 1e-7 * Math.pow(10, +$('softA').value / 100 * 5);
  sim.gravity = +$('grav').value;
  sim.tearEnabled = $('tear').checked;
  sim.tearStrain = +$('tearAt').value / 100;
}
['soft', 'softA', 'area', 'young', 'nu', 'grav', 'tear', 'cut', 'collide', 'tearAt'].forEach(id => $(id).addEventListener('input', applyControls));
$('fem').addEventListener('change', () => { applyControls(); build(); });
$('pin').addEventListener('change', build);
$('reset').addEventListener('click', build);
applyControls();
build();

// --- нож (M4): проведи по плите — вершины вдоль линии расщепляются перпендикулярно движению ---
let knife = false, stroke = null;
$('knife').addEventListener('click', () => {
  knife = !knife;
  $('knife').textContent = knife ? '✂ Нож: вкл' : '✂ Нож: выкл';
  $('knife').style.borderColor = knife ? '#ff6b6b' : '';
});
function knifeCut(ax, ay, bx, by, seen) {
  if (!sim.femEnabled) return;
  const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy);
  if (L < 1) return;
  const nx = -dy / L, ny = dx / L, n0 = sim.n;
  let any = false;
  for (let i = 0; i < n0; i++) {
    if (seen.has(i) || sim.w[i] === 0) continue;
    const px = sim.x[i * 2], py = sim.x[i * 2 + 1];
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (L * L)));
    if (Math.hypot(px - (ax + t * dx), py - (ay + t * dy)) < 9) {
      const nv = sim.splitVertex(i, nx, ny);
      seen.add(i);
      if (nv >= 0) { seen.add(nv); any = true; }
    }
  }
  if (any) sim.computePieces();
}

// --- перетаскивание мышью/пальцем: вершина становится кинематической ---
let drag = -1, dragW = 1;
function pointer(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
canvas.addEventListener('pointerdown', e => {
  const [mx, my] = pointer(e);
  if (knife) { stroke = { last: [mx, my], seen: new Set() }; canvas.setPointerCapture(e.pointerId); return; }
  const k = sim.pick(mx, my, 40);
  if (k >= 0) {
    drag = k; dragW = sim.w[k]; sim.w[k] = 0; canvas.setPointerCapture(e.pointerId);
    dragT = [sim.x[k * 2], sim.x[k * 2 + 1]]; dragV = [0, 0];
  }
});
canvas.addEventListener('pointermove', e => {
  if (stroke) {
    const [mx, my] = pointer(e);
    if (Math.hypot(mx - stroke.last[0], my - stroke.last[1]) > 6) {
      knifeCut(stroke.last[0], stroke.last[1], mx, my, stroke.seen);
      stroke.last = [mx, my];
    }
    return;
  }
  if (drag < 0) return;
  dragT = pointer(e); // цель; сама вершина едет к ней с ограничением скорости (см. moveDrag)
});

// Вершина под пальцем едет к цели не быстрее MAX_DRAG px/с: иначе кусок «телепортируется» сквозь другой
let dragT = null, dragV = [0, 0];
const MAX_DRAG = 1500;
function moveDrag(dt) {
  if (drag < 0 || !dragT) return;
  const dx = dragT[0] - sim.x[drag * 2], dy = dragT[1] - sim.x[drag * 2 + 1], L = Math.hypot(dx, dy);
  if (L < 1e-6) { dragV = [0, 0]; return; }
  const m = Math.min(L, MAX_DRAG * dt);
  sim.x[drag * 2] += dx / L * m; sim.x[drag * 2 + 1] += dy / L * m;
  dragV = [dx / L * m / dt, dy / L * m / dt];
}
// отпустили — вершина сохраняет скорость руки (можно «кинуть» кусок)
function release() {
  stroke = null;
  if (drag >= 0) {
    sim.w[drag] = dragW;
    sim.v[drag * 2] = dragV[0]; sim.v[drag * 2 + 1] = dragV[1];
    drag = -1;
  }
}
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);

// --- рендер ---
function strainColor(s) {
  const t = Math.min(1, s / 0.3);
  const hue = 210 - t * 210; // синий → красный
  return `hsl(${hue.toFixed(0)}, 75%, ${(38 + t * 12).toFixed(0)}%)`;
}

// M3: цвет по главному напряжению, нормированному на прочность (1 = момент разрыва)
function stressColor(r) {
  const t = Math.pow(Math.min(1, Math.max(0, r)), 0.6);
  return `hsl(${(210 - t * 210).toFixed(0)}, 75%, ${(38 + t * 12).toFixed(0)}%)`;
}
function stressPct() {
  let mx = 0;
  for (let t = 0; t < sim.sigma.length; t++) if (sim.sigma[t] > mx) mx = sim.sigma[t];
  return 100 * mx / (sim.tearStrain * sim.young);
}

function draw() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  const { x, tris, triEdges, alive, strain, sigma, triAlive } = sim;
  const crit = sim.tearStrain * sim.young;
  const T = tris.length / 3;

  for (let t = 0; t < T; t++) {
    const e0 = triEdges[t * 3], e1 = triEdges[t * 3 + 1], e2 = triEdges[t * 3 + 2];
    if (!triAlive[t] || !alive[e0] || !alive[e1] || !alive[e2]) continue; // наивно: треугольник с порванным ребром не рисуем
    const a = tris[t * 3], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
    ctx.beginPath();
    ctx.moveTo(x[a * 2], x[a * 2 + 1]);
    ctx.lineTo(x[b * 2], x[b * 2 + 1]);
    ctx.lineTo(x[c * 2], x[c * 2 + 1]);
    ctx.closePath();
    ctx.fillStyle = sim.femEnabled
      ? stressColor(sigma[t] / crit)
      : strainColor(Math.max(strain[e0], strain[e1], strain[e2]));
    ctx.fill();
    if (sim.pieces > 1 && sim.femEnabled) {
      ctx.strokeStyle = `hsla(${(sim.pieceId[t] * 47) % 360}, 70%, 72%, 0.6)`; // каждый кусок — свой оттенок рёбер
      ctx.lineWidth = 0.9;
    } else {
      ctx.strokeStyle = 'rgba(0,0,0,0.25)';
      ctx.lineWidth = 0.6;
    }
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
  while (acc >= DT) { moveDrag(DT); sim.step(DT); acc -= DT; }
  draw();
  $('info').textContent = `частиц: ${sim.n} · кусков: ${sim.pieces} · трещин: ${sim.splits} · контактов: ${sim.contacts} · вывернуто: ${sim.invertedCount()} · E: ${sim.energy().toFixed(0)}` + (sim.femEnabled ? ` · σmax: ${stressPct().toFixed(0)}% прочности` : '');
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
