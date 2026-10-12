// M6: обёртка над C++/WASM ядром. Класс WasmSim наследует Sim: интерфейс тот же, но step() выполняется в WebAssembly.
// Массивы состояния лежат в линейной памяти WASM, а поля sim.x, sim.v, sim.tris … — TypedArray-представления
// на эту память, поэтому разрез, перетаскивание и рендер (JS) работают с теми же данными без копирования.
import { Sim } from './sim.js';

// Порядок ОБЯЗАН совпадать с enum в wasm/core.cpp
const PN = ['GRAV', 'COMPL', 'ACOMPL', 'DAMP', 'YOUNG', 'NU', 'TEARSTRAIN', 'BW', 'BH', 'FRIC', 'CELL'];
const QN = ['N', 'T', 'M', 'CAP', 'SUBSTEPS', 'FEM', 'AREA', 'COLLIDE', 'TEAR', 'CUT', 'PIECES', 'CONTACTS',
  'GRIDCAP', 'ITEMSCAP', 'NCOLORS', 'SIMD',
  'X', 'P', 'V', 'W', 'EDGES', 'REST', 'ALIVE', 'STRAIN', 'TRIS', 'TRIEDGES', 'RESTAREA', 'DMINV', 'VOL',
  'TRIALIVE', 'SIGMA', 'SXX', 'SXY', 'SYY', 'PIECEID', 'VPIECE', 'CNT', 'POS', 'BB', 'ITEMS', 'BND',
  'BSTART', 'BCOUNT', 'CORDER', 'CSTART', 'CMASK'];
const P = {}, Q = {};
PN.forEach((k, i) => { P[k] = i; });
QN.forEach((k, i) => { Q[k] = i; });

// Поддерживает ли среда WASM SIMD (минимальный модуль с одной v128-инструкцией)
export function simdSupported() {
  try {
    return WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));
  } catch (e) { return false; }
}

// Загрузка модуля: bytes — ArrayBuffer/Uint8Array (Node) или URL (браузер)
export async function loadCore(src) {
  let bytes = src;
  if (typeof src === 'string') bytes = await (await fetch(src)).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance;
}

const align8 = n => (n + 7) & ~7;

export class WasmSim extends Sim {
  constructor(instance, opts = {}) {
    super();
    this.useColoring = !!opts.simd;
    this.inst = instance;
    this.ex = instance.exports;
    this.mem = this.ex.memory;
    this.P = new Float64Array(this.mem.buffer, this.ex.ctx_f64(), 32);
    this.Q = new Int32Array(this.mem.buffer, this.ex.ctx_i32(), 64);
    this.bound = false;
    this.gridCap = 0;
    this.itemsCap = 0;
    this.lastStepMs = 0;
  }

  // Размещаем все массивы в линейной памяти (после __heap_base), переносим в них содержимое и подменяем поля.
  // memory.grow делает старые представления недействительными — поэтому все представления создаются заново.
  _bind(gridNeed) {
    const names = {
      x: ['X', Float64Array], p: ['P', Float64Array], v: ['V', Float64Array], w: ['W', Float64Array],
      rest: ['REST', Float64Array], restArea: ['RESTAREA', Float64Array], dmInv: ['DMINV', Float64Array], vol: ['VOL', Float64Array],
      edges: ['EDGES', Int32Array], tris: ['TRIS', Int32Array], triEdges: ['TRIEDGES', Int32Array],
      pieceId: ['PIECEID', Int32Array], vPiece: ['VPIECE', Int32Array],
      alive: ['ALIVE', Uint8Array], triAlive: ['TRIALIVE', Uint8Array],
      strain: ['STRAIN', Float32Array], sigma: ['SIGMA', Float32Array],
      sxx: ['SXX', Float32Array], sxy: ['SXY', Float32Array], syy: ['SYY', Float32Array],
    };
    const T = this.triAlive.length, cap = this.cap;
    const snap = {};
    for (const k in names) snap[k] = this[k].slice();
    const gridCap = Math.max(gridNeed, 4096), itemsCap = Math.max(T * 16, 1 << 14);
    const extra = { CNT: gridCap + 1, POS: gridCap, BB: T * 4, ITEMS: itemsCap, BND: 9 * T + 3, CORDER: T + 4, CSTART: 66, BSTART: cap, BCOUNT: cap };
    let total = 0;
    for (const k in names) total += align8(snap[k].byteLength) + 8;
    for (const k in extra) total += align8(extra[k] * 4) + 8;
    let base = align8(this.ex.__heap_base.value);
    const need = base + total + 65536;
    const have = this.mem.buffer.byteLength;
    if (need > have) this.mem.grow(Math.ceil((need - have) / 65536));
    const buf = this.mem.buffer;
    this.P = new Float64Array(buf, this.ex.ctx_f64(), 32);
    this.Q = new Int32Array(buf, this.ex.ctx_i32(), 64);
    let off = base;
    for (const k in names) {
      const [qn, Ctor] = names[k];
      const arr = new Ctor(buf, off, snap[k].length);
      arr.set(snap[k]);
      this[k] = arr; this.Q[Q[qn]] = off;
      off += align8(snap[k].byteLength) + 8;
    }
    const ex = {};
    for (const k in extra) {
      ex[k] = new Int32Array(buf, off, extra[k]); this.Q[Q[k]] = off;
      off += align8(extra[k] * 4) + 8;
    }
    this._corder = ex.CORDER; this._cstart = ex.CSTART;
    this._bnd = ex.BND; this._bStart = ex.BSTART; this._bCount = ex.BCOUNT;
    this.gridCap = gridCap; this.itemsCap = itemsCap;
    this.Q[Q.GRIDCAP] = gridCap; this.Q[Q.ITEMSCAP] = itemsCap;
    this.Q[Q.NCOLORS] = 0; this.Q[Q.SIMD] = 0;
    this.bound = true;
    if (this.useColoring) this._recolor();
  }

  _gridCells() {
    const cs = this.cellSize;
    return (Math.ceil(this.bounds.w / cs) + 2) * (Math.ceil(this.bounds.h / cs) + 2);
  }

  buildSlab(nx, ny, spacing, ox, oy, pinLeft) {
    this.bound = false;
    super.buildSlab(nx, ny, spacing, ox, oy, pinLeft); // внутри computePieces → buildBoundary (пока без привязки)
    this._bind(this._gridCells() * 2);
    this.buildBoundary();
  }

  // Граничные рёбра кусков: плоский массив троек (a, b, элемент-владелец), у каждого куска — непрерывный диапазон
  buildBoundary() {
    super.buildBoundary();
    if (!this.bound) return;
    this._bCount.fill(0); this._bStart.fill(0);
    let off = 0;
    for (const [pc, arr] of this.bByPiece) {
      this._bStart[pc] = off; this._bCount[pc] = arr.length / 3;
      this._bnd.set(arr, off * 3);
      off += arr.length / 3;
    }
    if (this.useColoring) this._recolor(); // топология изменилась (вершины расщеплены)
  }

  // Жадная раскраска элементов: два элемента одного цвета не делят вершин → их можно решать одновременно (SIMD).
  _recolor() {
    const T = this.triAlive.length, tr = this.tris, vm = new Uint32Array(this.cap), col = new Int8Array(T);
    let nc = 0;
    for (let t = 0; t < T; t++) {
      const used = vm[tr[t * 3]] | vm[tr[t * 3 + 1]] | vm[tr[t * 3 + 2]];
      let c = 0; while (used & (1 << c)) c++;
      if (c >= 32) { this.Q[Q.NCOLORS] = 0; return; } // не должно случаться на сетках из треугольников
      col[t] = c; if (c + 1 > nc) nc = c + 1;
      const bit = 1 << c; vm[tr[t * 3]] |= bit; vm[tr[t * 3 + 1]] |= bit; vm[tr[t * 3 + 2]] |= bit;
    }
    this.colors = nc;
    const cnt = new Int32Array(nc + 1);
    for (let t = 0; t < T; t++) cnt[col[t] + 1]++;
    for (let c = 0; c < nc; c++) cnt[c + 1] += cnt[c];
    this._cstart.set(cnt);
    const pos = cnt.slice();
    for (let t = 0; t < T; t++) this._corder[pos[col[t]]++] = t;
    this.Q[Q.NCOLORS] = nc;
  }

  step(dt) {
    if (!this.bound) return;
    const t0 = performance.now();
    if (this._gridCells() > this.gridCap) this._bind(this._gridCells() * 2); // окно выросло — пересоздаём сетку
    const P_ = this.P, Q_ = this.Q;
    P_[P.GRAV] = this.gravity; P_[P.COMPL] = this.compliance; P_[P.ACOMPL] = this.areaCompliance; P_[P.DAMP] = this.damping;
    P_[P.YOUNG] = this.young; P_[P.NU] = this.nu; P_[P.TEARSTRAIN] = this.tearStrain;
    P_[P.BW] = this.bounds.w; P_[P.BH] = this.bounds.h; P_[P.FRIC] = this.friction; P_[P.CELL] = this.cellSize;
    Q_[Q.N] = this.n; Q_[Q.T] = this.triAlive.length; Q_[Q.M] = this.rest.length; Q_[Q.CAP] = this.cap;
    Q_[Q.SUBSTEPS] = this.substeps; Q_[Q.FEM] = +this.femEnabled; Q_[Q.AREA] = +this.areaEnabled;
    Q_[Q.COLLIDE] = +this.collideEnabled; Q_[Q.TEAR] = +this.tearEnabled; Q_[Q.CUT] = +this.cutEnabled;
    Q_[Q.PIECES] = this.pieces;
    this.ex.step(dt);
    this.contacts = Q_[Q.CONTACTS];
    if (this.femEnabled && this.tearEnabled && this.cutEnabled) this.cutStep(this.tearStrain * this.young);
    this.lastStepMs = performance.now() - t0;
  }
}
