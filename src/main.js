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
  // плита стартует ниже панели управления (на телефоне панель занимает несколько строк)
  const oy = Math.max(140, Math.ceil($('hud').getBoundingClientRect().bottom) + 20);
  sim.buildSlab(nx, ny, spacing, ox, oy, $('pin').checked);
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
  showValues();
}

// Числа рядом с ползунками: крупно — положение ползунка (его называют в инструкциях),
// бледно в скобках — реальное значение в физических единицах.
const sci = v => v.toExponential(1).replace('.0e', 'e').replace('e+', 'e');
function showValues() {
  const set = (id, pos, phys) => { $('v-' + id).innerHTML = phys ? `${pos} <i>(${phys})</i>` : pos; };
  set('soft', $('soft').value, sci(sim.compliance));
  set('softA', $('softA').value, sci(sim.areaCompliance));
  set('young', $('young').value, sci(sim.young));
  set('nu', $('nu').value, sim.nu.toFixed(2));
  set('grav', $('grav').value);
  set('tearAt', $('tearAt').value, sim.femEnabled ? '% от E' : '% удл.');
}
['soft', 'softA', 'area', 'young', 'nu', 'grav', 'tear', 'cut', 'collide', 'tearAt'].forEach(id => $(id).addEventListener('input', applyControls));
$('fem').addEventListener('change', () => { applyControls(); build(); });
$('pin').addEventListener('change', build);
$('reset').addEventListener('click', build);
applyControls();
build();

// --- нож (M4.1): проведи линию по плите и отпусти — плита разрежется вдоль неё ---
let knife = false, stroke = null;
$('knife').addEventListener('click', () => {
  knife = !knife;
  $('knife').textContent = knife ? '✂ Нож: вкл' : '✂ Нож: выкл';
  $('knife').style.borderColor = knife ? '#ff6b6b' : '';
});
// M4.1: нож рисует линию (красный след), разрез делается при отпускании пальца/кнопки:
// вершины делятся по стороне от линии, любой угол и изгиб, без «полосок» (см. sim.cutAlong)
function endStroke(doCut) {
  const s = stroke; stroke = null;
  if (!s || !doCut || !sim.femEnabled || s.pts.length < 4) return;
  const L = s.pts.reduce((a, _, i) => i >= 2 && i % 2 === 0 ? a + Math.hypot(s.pts[i] - s.pts[i - 2], s.pts[i + 1] - s.pts[i - 1]) : a, 0);
  if (L >= 10) sim.cutAlong(s.pts, 3.5 * sim.spacing);
}

// --- перетаскивание мышью/пальцем: вершина становится кинематической ---
let drag = -1, dragW = 1;
function pointer(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
canvas.addEventListener('pointerdown', e => {
  const [mx, my] = pointer(e);
  if (knife) { stroke = { pts: [mx, my] }; canvas.setPointerCapture(e.pointerId); return; }
  const k = sim.pick(mx, my, 40);
  if (k >= 0) {
    drag = k; dragW = sim.w[k]; sim.w[k] = 0; canvas.setPointerCapture(e.pointerId);
    dragT = [sim.x[k * 2], sim.x[k * 2 + 1]]; dragV = [0, 0];
  }
});
canvas.addEventListener('pointermove', e => {
  if (stroke) {
    const [mx, my] = pointer(e);
    const p = stroke.pts, n = p.length;
    if (Math.hypot(mx - p[n - 2], my - p[n - 1]) >= 4) p.push(mx, my);
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
function release(e) {
  endStroke(!(e && e.type === 'pointercancel'));
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

  // след ножа
  if (stroke && stroke.pts.length >= 4) {
    ctx.strokeStyle = '#ff6b6b'; ctx.lineWidth = 2.5; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.setLineDash([7, 5]);
    ctx.beginPath(); ctx.moveTo(stroke.pts[0], stroke.pts[1]);
    for (let i = 2; i < stroke.pts.length; i += 2) ctx.lineTo(stroke.pts[i], stroke.pts[i + 1]);
    ctx.stroke(); ctx.setLineDash([]);
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
