// Бенчмарк: JS vs WASM (scalar / simd). Запуск: node bench/bench.mjs
import fs from 'fs';
import { Sim } from '../src/sim.js';
import { WasmSim, loadCore, simdSupported } from '../src/wasm-sim.js';
const dir = new URL('../src/wasm/', import.meta.url);
const variants = { js: null };
variants.scalar = await loadCore(fs.readFileSync(new URL('core.scalar.wasm', dir)));
if (simdSupported() && fs.existsSync(new URL('core.simd.wasm', dir))) variants.simd = await loadCore(fs.readFileSync(new URL('core.simd.wasm', dir)));
const sizes = [[26, 8], [50, 20], [100, 40], [200, 80]];
const FR = 120;
console.log('сетка (треуг.)      ' + Object.keys(variants).map(k => k.padStart(10)).join('') + '   (мс/кадр, медиана; 10 подшагов)');
for (const [nx, ny] of sizes) {
  const row = [];
  for (const k in variants) {
    const s = k === 'js' ? new Sim() : new WasmSim(variants[k], { simd: k === 'simd' });
    s.bounds.w = 4000; s.bounds.h = 2000; s.collideEnabled = false;
    s.buildSlab(nx, ny, 12, 100, 100, true);
    for (let i = 0; i < 20; i++) s.step(1 / 60); // прогрев JIT
    const ts = [];
    for (let i = 0; i < FR; i++) { const t = performance.now(); s.step(1 / 60); ts.push(performance.now() - t); }
    ts.sort((a, b) => a - b); row.push(ts[FR >> 1].toFixed(3).padStart(10));
  }
  console.log(`${nx}x${ny} (${nx * ny * 2})`.padEnd(20) + row.join(''));
}
