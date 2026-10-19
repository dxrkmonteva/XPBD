// M8: 3D-демо. Физика — Sim3D (JS), рендер — Renderer3D (WebGL2).
// Управление: тянуть тело — захват вершины; тянуть пустое место — вращение камеры; колесо / щипок — масштаб.
import { Sim3D } from './sim3d.js';
import { Sim3DWasm, loadCore3D } from './wasm-sim3d.js';
import { simdSupported } from './wasm-sim.js';
import scalarUrl from './wasm/core3d.scalar.wasm?url';
import simdUrl from './wasm/core3d.simd.wasm?url';
import { Renderer3D, Camera } from './gl3d.js';

const $ = id => document.getElementById(id);
const canvas = $('c');
let gl3d;
try { gl3d = new Renderer3D(canvas); } catch (e) { $('err').textContent = String(e.message || e); throw e; }
const cam = new Camera();
let sim = new Sim3D();
const cores = {};
async function makeSim(kind) {
  if (kind === 'js') return new Sim3D();
  try {
    if (!cores[kind]) cores[kind] = await loadCore3D(kind === 'simd' ? simdUrl : scalarUrl);
    return new Sim3DWasm(cores[kind], { simd: kind === 'simd' });
  } catch (e) { $('err').textContent = 'WASM недоступен: ' + (e.message || e); return new Sim3D(); }
}
if (!simdSupported()) $('backend').querySelector('option[value=simd]').disabled = true, $('backend').value = 'scalar';
async function setBackend() { sim = await makeSim($('backend').value); build(); }
$('backend').addEventListener('change', setBackend);

function build() {
  const k = +$('detail').value;
  sim.buildBox(8 * k, 2 * k, 2 * k, 0.25 / k, 0, 0.6, 0);
  gl3d.setMesh(sim);
  applyControls();
}

function applyControls() {
  sim.young = 1e5 * Math.pow(10, 2.3 * $('young').value / 100); // 1e5 … 2e7 Па
  sim.nu = +$('nu').value / 100;
  sim.gravity = +$('grav').value;
  sim.substeps = +$('sub').value;
  sim.friction = +$('fric').value / 100;
  gl3d.colorize = $('colorize').checked ? 1 / 0.5 : 0;
  sim.pinMinX($('pin').checked, 1e-6);
  $('v-young').textContent = (sim.young / 1e6).toFixed(2) + ' МПа';
  $('v-nu').textContent = sim.nu.toFixed(2);
  $('v-grav').textContent = sim.gravity.toFixed(1);
  $('v-sub').textContent = sim.substeps;
  $('v-fric').textContent = sim.friction.toFixed(2);
  $('v-detail').textContent = '×' + $('detail').value;
}
['young', 'nu', 'grav', 'sub', 'fric', 'pin', 'colorize'].forEach(id => $(id).addEventListener('input', applyControls));
$('detail').addEventListener('input', () => { $('v-detail').textContent = '×' + $('detail').value; });
$('detail').addEventListener('change', build);
$('reset').addEventListener('click', build);

// ---- ввод: указатели ----
const ptrs = new Map(); let mode = null, grabDepth = 0, pinchD = 0;
const aspect = () => canvas.clientWidth / canvas.clientHeight;
function rayAt(e) { const r = canvas.getBoundingClientRect(); return cam.ray(((e.clientX - r.left) / r.width) * 2 - 1, 1 - ((e.clientY - r.top) / r.height) * 2, aspect()); }
function dragTarget(e) { // пересечение луча с плоскостью, перпендикулярной взгляду, на глубине захвата
  const ray = rayAt(e), f = ray.f, t = grabDepth / (ray.d[0] * f[0] + ray.d[1] * f[1] + ray.d[2] * f[2]);
  return [ray.o[0] + ray.d[0] * t, Math.max(0.02, ray.o[1] + ray.d[1] * t), ray.o[2] + ray.d[2] * t];
}
canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (ptrs.size === 2) { sim.release(); mode = 'pinch'; const [a, b] = [...ptrs.values()]; pinchD = Math.hypot(a.x - b.x, a.y - b.y); return; }
  const ray = rayAt(e), hit = gl3d.pick(sim, ray);
  if (hit) { mode = 'grab'; grabDepth = hit.t * (ray.d[0] * ray.f[0] + ray.d[1] * ray.f[1] + ray.d[2] * ray.f[2]); sim.grab(hit.vertex, ...hit.point); }
  else mode = 'orbit';
});
canvas.addEventListener('pointermove', e => {
  const p = ptrs.get(e.pointerId); if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
  if (mode === 'grab' && ptrs.size === 1) { const t = dragTarget(e); sim.grab(sim.drag.v, ...t); }
  else if (mode === 'orbit' && ptrs.size === 1) { cam.yaw -= dx * 0.008; cam.pitch = Math.max(-0.2, Math.min(1.45, cam.pitch + dy * 0.008)); }
  else if (mode === 'pinch' && ptrs.size === 2) { const [a, b] = [...ptrs.values()], d = Math.hypot(a.x - b.x, a.y - b.y); if (pinchD > 0) cam.dist = Math.max(1.2, Math.min(14, cam.dist * pinchD / d)); pinchD = d; }
});
const end = e => { ptrs.delete(e.pointerId); if (ptrs.size === 0) { sim.release(); mode = null; } else if (ptrs.size === 1) { mode = 'orbit'; sim.release(); } };
canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
canvas.addEventListener('wheel', e => { e.preventDefault(); cam.dist = Math.max(1.2, Math.min(14, cam.dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });

// ---- главный цикл: один фиксированный шаг 1/60 за кадр ----
let ms = 0, frames = 0;
function loop() {
  const t0 = performance.now();
  sim.step(1 / 60);
  const t1 = performance.now();
  gl3d.draw(sim, cam);
  const t2 = performance.now();
  ms = ms * 0.9 + (t1 - t0) * 0.1;
  if (++frames % 15 === 0) {
    $('info').textContent = `ядро: ${$('backend').value} · физика ${ms.toFixed(1)} мс/кадр · тетраэдров: ${sim.T} · вершин: ${sim.n} · вывернуто: ${sim.invertedCount()} · объём: ${(sim.volumeRatio() * 100).toFixed(1)}% · рендер ${(t2 - t1).toFixed(1)} мс`;
  }
  requestAnimationFrame(loop);
}
setBackend().then(() => requestAnimationFrame(loop));
window.__dbg = { get sim() { return sim; }, gl3d, cam };
