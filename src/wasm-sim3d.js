// M8: обёртка над C++/WASM 3D-ядром (wasm/core3d.cpp). Sim3DWasm наследует Sim3D: интерфейс тот же, step() — в WebAssembly.
// Массивы состояния лежат в линейной памяти WASM, поля sim.x, sim.v, sim.tets … — TypedArray-представления на неё.
import { Sim3D } from './sim3d.js';

// Порядок ОБЯЗАН совпадать с enum в wasm/core3d.cpp
const PN = ['GRAV', 'YOUNG', 'NU', 'DAMP', 'FRIC', 'FLOORY', 'DCOMPL', 'DX', 'DY', 'DZ'];
const QN = ['N', 'T', 'SUBSTEPS', 'DRAGV', 'NCOLORS', 'SIMD', 'DIRTY', 'TP',
  'X', 'P', 'V', 'W', 'TETS', 'DMINV', 'REST', 'QUAT', 'DEFO', 'CORDER', 'CSTART', 'ORD', 'SOA'];
const P = {}, Q = {};
PN.forEach((k, i) => { P[k] = i; });
QN.forEach((k, i) => { Q[k] = i; });
const MAX_COLORS = 128, SOA_ARRAYS = 23; // число массивов SoA — O_N в core3d.cpp
const align8 = n => (n + 7) & ~7;

export async function loadCore3D(src) {
  let bytes = src;
  if (typeof src === 'string') bytes = await (await fetch(src)).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance;
}

export class Sim3DWasm extends Sim3D {
  constructor(instance, opts = {}) {
    super();
    this.simd = !!opts.simd;
    this.ex = instance.exports;
    this.mem = this.ex.memory;
    this.P = new Float64Array(this.mem.buffer, this.ex.ctx_f64(), 16);
    this.Q = new Int32Array(this.mem.buffer, this.ex.ctx_i32(), 48);
    this.bound = false;
    this.ncolors = 0;
  }

  buildBox(nx, ny, nz, h, ox, oy, oz) {
    this.bound = false;
    super.buildBox(nx, ny, nz, h, ox, oy, oz);
    this._bind();
  }

  _bind() {
    const T = this.T, n = this.n;
    const names = {
      x: ['X', Float64Array], p: ['P', Float64Array], v: ['V', Float64Array], w: ['W', Float64Array],
      tets: ['TETS', Int32Array], dmInv: ['DMINV', Float64Array], rest: ['REST', Float64Array], quat: ['QUAT', Float64Array], defo: ['DEFO', Float32Array],
    };
    const snap = {};
    for (const k in names) snap[k] = this[k].slice();
    const Tp = T + 4 * MAX_COLORS;
    const extra = { CORDER: [T, Int32Array], CSTART: [MAX_COLORS + 2, Int32Array], ORD: [Tp, Int32Array], SOA: [SOA_ARRAYS * Tp, Float32Array] };
    let total = 0;
    for (const k in names) total += align8(snap[k].byteLength) + 8;
    for (const k in extra) total += align8(extra[k][0] * 4) + 8;
    const base = align8(this.ex.__heap_base.value);
    const need = base + total + 65536, have = this.mem.buffer.byteLength;
    if (need > have) this.mem.grow(Math.ceil((need - have) / 65536));
    const buf = this.mem.buffer;
    this.P = new Float64Array(buf, this.ex.ctx_f64(), 16);
    this.Q = new Int32Array(buf, this.ex.ctx_i32(), 48);
    let off = base;
    for (const k in names) {
      const [qn, Ctor] = names[k];
      const arr = new Ctor(buf, off, snap[k].length);
      arr.set(snap[k]);
      this[k] = arr; this.Q[Q[qn]] = off;
      off += align8(snap[k].byteLength) + 8;
    }
    for (const k in extra) {
      const [len, Ctor] = extra[k];
      this['_' + k.toLowerCase()] = new Ctor(buf, off, len); this.Q[Q[k]] = off;
      off += align8(len * 4) + 8;
    }
    this.Q[Q.N] = n; this.Q[Q.T] = T; this.Q[Q.SIMD] = 0; this.Q[Q.NCOLORS] = 0;
    this.bound = true;
    if (this.simd) this._recolor();
  }

  // Жадная раскраска тетраэдров: общих вершин у элементов одного цвета нет. Порядок: цвет за цветом, внутри цвета по номеру
  // элемента (пространственная когерентность), каждый цвет дополнен пустыми слотами до кратного 4.
  _recolor() {
    const T = this.T, n = this.n, tets = this.tets, W = 4;
    const used = new Uint32Array(n * W), col = new Int32Array(T);
    let nc = 0;
    for (let t = 0; t < T; t++) {
      const a = tets[t * 4], b = tets[t * 4 + 1], c = tets[t * 4 + 2], d = tets[t * 4 + 3];
      let k = 0;
      for (;; k++) {
        const wd = k >> 5, bit = 1 << (k & 31);
        if (!((used[a * W + wd] | used[b * W + wd] | used[c * W + wd] | used[d * W + wd]) & bit)) break;
      }
      if (k >= MAX_COLORS) throw new Error('слишком много цветов раскраски');
      col[t] = k; if (k + 1 > nc) nc = k + 1;
      const wd = k >> 5, bit = 1 << (k & 31);
      used[a * W + wd] |= bit; used[b * W + wd] |= bit; used[c * W + wd] |= bit; used[d * W + wd] |= bit;
    }
    const cnt = new Int32Array(nc);
    for (let t = 0; t < T; t++) cnt[col[t]]++;
    const start = this._cstart, ord = this._ord;
    let s = 0;
    for (let c = 0; c < nc; c++) { start[c] = s; s += (cnt[c] + 3) & ~3; }
    start[nc] = s;
    ord.fill(-1);
    const pos = Int32Array.from(start.subarray(0, nc));
    for (let t = 0; t < T; t++) ord[pos[col[t]]++] = t;
    this.ncolors = nc;
    this.Q[Q.TP] = s; this.Q[Q.NCOLORS] = nc; this.Q[Q.SIMD] = 1; this.Q[Q.DIRTY] = 1;
  }

  step(dt) {
    if (!this.bound) return;
    const P_ = this.P, Q_ = this.Q, d = this.drag;
    P_[P.GRAV] = this.gravity; P_[P.YOUNG] = this.young; P_[P.NU] = this.nu; P_[P.DAMP] = this.damping; P_[P.FRIC] = this.friction;
    P_[P.FLOORY] = this.floorY; P_[P.DCOMPL] = d.compliance; P_[P.DX] = d.tx; P_[P.DY] = d.ty; P_[P.DZ] = d.tz;
    Q_[Q.SUBSTEPS] = this.substeps; Q_[Q.DRAGV] = d.v;
    this.ex.step3(dt);
  }
}
