// M8: 3D-демо. Физика — Sim3D (JS), рендер — Renderer3D (WebGL2).
// Управление: тянуть тело — захват вершины; тянуть пустое место — вращение камеры; колесо / щипок — масштаб.
import { Sim3D } from './sim3d.js';
import { Sim3DWasm, loadCore3D } from './wasm-sim3d.js';
import { simdSupported } from './wasm-sim.js';
import scalarUrl from './wasm/core3d.scalar.wasm?url';
import simdUrl from './wasm/core3d.simd.wasm?url';
import { Renderer3D, Camera } from './gl3d.js';
import { Sim3DGPU, createGPU, gpuAvailable } from './gpusim3d.js';

const $ = id => document.getElementById(id);
const canvas = $('c');
let gl3d;
try { gl3d = new Renderer3D(canvas); } catch (e) { $('err').textContent = String(e.message || e); throw e; }
const cam = new Camera();
let sim = new Sim3D();
const cores = {};
let gpuCtx = null;
async function makeSim(kind) {
  if (kind === 'js') return new Sim3D();
  if (kind === 'gpu') {
    try {
      if (!gpuCtx) gpuCtx = await createGPU();
      const k = simdSupported() ? 'simd' : 'scalar';
      if (!cores[k]) cores[k] = await loadCore3D(k === 'simd' ? simdUrl : scalarUrl);
      $('err').textContent = '';
      return new Sim3DGPU(cores[k], gpuCtx);
    } catch (e) { $('err').textContent = 'GPU недоступен: ' + (e.message || e); $('backend').value = 'simd'; kind = 'simd'; }
  }
  try {
    if (!cores[kind]) cores[kind] = await loadCore3D(kind === 'simd' ? simdUrl : scalarUrl);
    return new Sim3DWasm(cores[kind], { simd: kind === 'simd' });
  } catch (e) { $('err').textContent = 'WASM недоступен: ' + (e.message || e); return new Sim3D(); }
}
if (!gpuAvailable()) $('backend').querySelector('option[value=gpu]').disabled = true;
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
  gl3d.colorize = $('colorize').checked ? 1 / 0.5 : 0; sim.wantDefo = $('colorize').checked;
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
const ptrs = new Map(); let mode = null, pinchD = 0; // ptrs: id → {x, y, grab?: глубина захвата}
const aspect = () => canvas.clientWidth / canvas.clientHeight;
function rayAt(e) { const r = canvas.getBoundingClientRect(); return cam.ray(((e.clientX - r.left) / r.width) * 2 - 1, 1 - ((e.clientY - r.top) / r.height) * 2, aspect()); }
function dragTarget(e, depth) { // пересечение луча с плоскостью, перпендикулярной взгляду, на глубине захвата
  const ray = rayAt(e), f = ray.f, t = depth / (ray.d[0] * f[0] + ray.d[1] * f[1] + ray.d[2] * f[2]);
  return [ray.o[0] + ray.d[0] * t, Math.max(0.02, ray.o[1] + ray.d[1] * t), ray.o[2] + ray.d[2] * t];
}
const grabsActive = () => [...ptrs.values()].filter(p => p.grab !== undefined).length;
canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId);
  const p = { x: e.clientX, y: e.clientY }; ptrs.set(e.pointerId, p);
  const ray = rayAt(e), hit = gl3d.pick(sim, ray);
  if (hit) { // каждый палец, попавший в тело, держит свою область — можно тянуть двумя пальцами в разные стороны
    p.grab = hit.t * (ray.d[0] * ray.f[0] + ray.d[1] * ray.f[1] + ray.d[2] * ray.f[2]);
    sim.startGrab(hit.point[0], hit.point[1], hit.point[2], Math.max(0.15, 3 * sim.spacing), ...hit.point, e.pointerId);
    mode = 'grab';
  } else if (mode === 'grab') { /* второй палец мимо тела при активном захвате — игнорируем */ }
  else if (ptrs.size === 2) { mode = 'pinch'; const [a, b] = [...ptrs.values()]; pinchD = Math.hypot(a.x - b.x, a.y - b.y); }
  else mode = 'orbit';
});
canvas.addEventListener('pointermove', e => {
  const p = ptrs.get(e.pointerId); if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
  if (p.grab !== undefined) sim.moveGrab(...dragTarget(e, p.grab), e.pointerId);
  else if (mode === 'orbit' && ptrs.size === 1) { cam.yaw -= dx * 0.008; cam.pitch = Math.max(-0.2, Math.min(1.45, cam.pitch + dy * 0.008)); }
  else if (mode === 'pinch' && ptrs.size === 2) { const [a, b] = [...ptrs.values()], d = Math.hypot(a.x - b.x, a.y - b.y); if (pinchD > 0) cam.dist = Math.max(1.2, Math.min(14, cam.dist * pinchD / d)); pinchD = d; }
});
const end = e => {
  const p = ptrs.get(e.pointerId); if (!p) return;
  if (p.grab !== undefined) sim.release(e.pointerId);
  ptrs.delete(e.pointerId);
  if (ptrs.size === 0) mode = null;
  else if (grabsActive() === 0) mode = 'orbit';
};
canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
canvas.addEventListener('wheel', e => { e.preventDefault(); cam.dist = Math.max(1.2, Math.min(14, cam.dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });

// ---- главный цикл: фиксированный шаг 1/60 с накоплением реального времени (допуск 15%, чтобы на 60 Гц не терять шаги из-за дрожания кадров) ----
let ms = 0, frames = 0, acc = 0, tPrev = performance.now();
const DT = 1 / 60, MAX_STEPS = 3;
function loop() {
  const t0 = performance.now();
  // фиксированный шаг по реальному времени: скорость физики не зависит от частоты экрана (60/90/120 Гц)
  acc += Math.min(0.05, (t0 - tPrev) / 1000); tPrev = t0;
  for (let k = 0; k < MAX_STEPS && acc >= DT * 0.85; k++) { if (sim.step(DT) === false) { acc = 0; break; } acc -= DT; }
  acc = Math.min(acc, DT); // долг не копим: если не успеваем, физика идёт медленнее реального времени, а не «спиралью»
  const t1 = performance.now();
  gl3d.draw(sim, cam);
  const t2 = performance.now();
  ms = ms * 0.9 + (t1 - t0) * 0.1;
  if (++frames % 15 === 0) {
    $('info').textContent = `ядро: ${$('backend').value} · ${sim.gpu ? 'GPU ' + (sim.gpuMs ? sim.gpuMs.toFixed(1) + ' мс чистых' : '—') + ' · кадр→готово ' + sim.statMs.toFixed(1) + ' · шаг каждые ' + sim.stepMs.toFixed(1) + ' мс · пропущено ' + sim.skipped : 'физика ' + ms.toFixed(1) + ' мс/кадр'} · тетраэдров: ${sim.T} · вершин: ${sim.n} · вывернуто: ${sim.invertedCount()} · объём: ${(sim.volumeRatio() * 100).toFixed(1)}% · рендер ${(t2 - t1).toFixed(1)} мс`;
  }
  requestAnimationFrame(loop);
}
setBackend().then(() => requestAnimationFrame(loop));
window.__dbg = { get sim() { return sim; }, gl3d, cam };
