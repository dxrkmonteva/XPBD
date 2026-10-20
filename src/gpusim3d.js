// M9: физика 3D на GPU (WebGPU compute). Тот же шаг XPBD-FEM, что в wasm/core3d.cpp (раскраска, подшаги), но в WGSL.
// Состояние живёт в буферах GPU; позиции асинхронно копируются обратно в sim.x (с задержкой 1–2 кадра) для рендера WebGL2.
// Mesh/раскраска/захваты — от Sim3DWasm; GPU-класс заменяет только step().
import { Sim3DWasm } from './wasm-sim3d.js';
import { DRAG_CAP } from './sim3d.js';

const WGSL = /* wgsl */`
struct Slot { count: u32, off: u32, n: u32, pad: u32, h: f32, ih2: f32, mu: f32, lam: f32, grav: f32, damp: f32, fric: f32, floorY: f32, dcomp: f32 };
@group(0) @binding(0) var<storage, read_write> pos: array<f32>;
@group(0) @binding(1) var<storage, read_write> prv: array<f32>;
@group(0) @binding(2) var<storage, read_write> vel: array<f32>;
@group(0) @binding(3) var<storage, read> wgt: array<f32>;
@group(0) @binding(4) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> dm: array<f32>;
@group(0) @binding(6) var<storage, read> invV: array<f32>;
@group(0) @binding(7) var<storage, read_write> quat: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> S: Slot;
@group(0) @binding(9) var<storage, read_write> defo: array<f32>;
@group(0) @binding(10) var<storage, read> drag: array<vec4<f32>>;

fn ld(i: u32) -> vec3<f32> { return vec3<f32>(pos[i * 3u], pos[i * 3u + 1u], pos[i * 3u + 2u]); }
fn st_add(i: u32, d: vec3<f32>) { pos[i * 3u] += d.x; pos[i * 3u + 1u] += d.y; pos[i * 3u + 2u] += d.z; }

@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= S.n) { return; }
  let a = i * 3u;
  prv[a] = pos[a]; prv[a + 1u] = pos[a + 1u]; prv[a + 2u] = pos[a + 2u];
  if (wgt[i] > 0.0) {
    vel[a + 1u] -= S.grav * S.h;
    pos[a] += vel[a] * S.h; pos[a + 1u] += vel[a + 1u] * S.h; pos[a + 2u] += vel[a + 2u] * S.h;
  }
}

@compute @workgroup_size(64)
fn post(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= S.n) { return; }
  let a = i * 3u;
  if (wgt[i] == 0.0) { vel[a] = 0.0; vel[a + 1u] = 0.0; vel[a + 2u] = 0.0; return; }
  if (pos[a + 1u] < S.floorY) {
    pos[a + 1u] = S.floorY;
    pos[a] = prv[a] + (pos[a] - prv[a]) * (1.0 - S.fric);
    pos[a + 2u] = prv[a + 2u] + (pos[a + 2u] - prv[a + 2u]) * (1.0 - S.fric);
  }
  let k = S.damp / S.h;
  vel[a] = (pos[a] - prv[a]) * k; vel[a + 1u] = (pos[a + 1u] - prv[a + 1u]) * k; vel[a + 2u] = (pos[a + 2u] - prv[a + 2u]) * k;
}

struct Del { a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32> };
fn deltaC(g0: vec3<f32>, g1: vec3<f32>, g2: vec3<f32>, Cv: f32, alpha: f32, m0: vec3<f32>, m1: vec3<f32>, m2: vec3<f32>, w: vec4<f32>) -> Del {
  let h1 = g0 * m0.x + g1 * m0.y + g2 * m0.z;
  let h2 = g0 * m1.x + g1 * m1.y + g2 * m1.z;
  let h3 = g0 * m2.x + g1 * m2.y + g2 * m2.z;
  let h0 = -(h1 + h2 + h3);
  let den = w.x * dot(h0, h0) + w.y * dot(h1, h1) + w.z * dot(h2, h2) + w.w * dot(h3, h3) + alpha;
  var r: Del;
  if (den < 1e-30) { r.a = vec3<f32>(0.0); r.b = r.a; r.c = r.a; r.d = r.a; return r; }
  let dl = -Cv / den;
  r.a = w.x * dl * h0; r.b = w.y * dl * h1; r.c = w.z * dl * h2; r.d = w.w * dl * h3;
  return r;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x; if (k >= S.count) { return; }
  let t = S.off + k;
  let tv = tets[t];
  let x0 = ld(tv.x);
  let d1 = ld(tv.y) - x0; let d2 = ld(tv.z) - x0; let d3 = ld(tv.w) - x0;
  let w = vec4<f32>(wgt[tv.x], wgt[tv.y], wgt[tv.z], wgt[tv.w]);
  let b = t * 9u;
  let m0 = vec3<f32>(dm[b], dm[b + 1u], dm[b + 2u]);
  let m1 = vec3<f32>(dm[b + 3u], dm[b + 4u], dm[b + 5u]);
  let m2 = vec3<f32>(dm[b + 6u], dm[b + 7u], dm[b + 8u]);
  let f0 = d1 * m0.x + d2 * m1.x + d3 * m2.x;
  let f1 = d1 * m0.y + d2 * m1.y + d3 * m2.y;
  let f2 = d1 * m0.z + d2 * m1.z + d3 * m2.z;
  var q = quat[t];
  var r0 = vec3<f32>(1.0, 0.0, 0.0); var r1 = vec3<f32>(0.0, 1.0, 0.0); var r2 = vec3<f32>(0.0, 0.0, 1.0);
  for (var it = 0; it < 3; it++) {
    let xx = q.x * q.x; let yy = q.y * q.y; let zz = q.z * q.z; let xy = q.x * q.y; let xz = q.x * q.z; let yz = q.y * q.z;
    let wx = q.w * q.x; let wy = q.w * q.y; let wz = q.w * q.z;
    r0 = vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy));
    r1 = vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx));
    r2 = vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy));
    if (it == 2) { break; }
    let om = (cross(r0, f0) + cross(r1, f1) + cross(r2, f2)) / (abs(dot(r0, f0) + dot(r1, f1) + dot(r2, f2)) + 1e-9);
    let s = om * 0.5;
    let n = vec4<f32>(q.x + s.x * q.w + s.y * q.z - s.z * q.y, q.y + s.y * q.w + s.z * q.x - s.x * q.z,
                      q.z + s.z * q.w + s.x * q.y - s.y * q.x, q.w - s.x * q.x - s.y * q.y - s.z * q.z);
    q = n * inverseSqrt(dot(n, n));
  }
  quat[t] = q;
  let e0 = f0 - r0; let e1 = f1 - r1; let e2 = f2 - r2;
  let Cd = sqrt(dot(e0, e0) + dot(e1, e1) + dot(e2, e2));
  defo[t] = Cd;
  var dA = vec3<f32>(0.0); var dB = dA; var dC = dA; var dD = dA;
  var g0 = vec3<f32>(0.0); var g1 = g0; var g2 = g0;
  var nd1 = d1; var nd2 = d2; var nd3 = d3;
  let iv = invV[t];
  if (Cd > 1e-6) {
    let ic = 1.0 / Cd;
    let dl = deltaC(e0 * ic, e1 * ic, e2 * ic, Cd, 0.5 / S.mu * iv * S.ih2, m0, m1, m2, w);
    dA = dl.a; dB = dl.b; dC = dl.c; dD = dl.d;
    nd1 = d1 + dB - dA; nd2 = d2 + dC - dA; nd3 = d3 + dD - dA;
  }
  let a0 = nd1 * m0.x + nd2 * m1.x + nd3 * m2.x;
  let a1 = nd1 * m0.y + nd2 * m1.y + nd3 * m2.y;
  let a2 = nd1 * m0.z + nd2 * m1.z + nd3 * m2.z;
  let det = dot(a0, cross(a1, a2));
  if (S.lam > 0.0 && abs(det - 1.0) > 1e-7) {
    let dl = deltaC(cross(a1, a2), cross(a2, a0), cross(a0, a1), det - 1.0, iv / S.lam * S.ih2, m0, m1, m2, w);
    dA += dl.a; dB += dl.b; dC += dl.c; dD += dl.d;
  }
  st_add(tv.x, dA); st_add(tv.y, dB); st_add(tv.z, dC); st_add(tv.w, dD);
}

@compute @workgroup_size(64)
fn dragk(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x; if (k >= S.count) { return; }
  let e = drag[k]; let i = bitcast<u32>(e.w); let wi = wgt[i]; if (wi <= 0.0) { return; }
  let d = ld(i) - e.xyz; let L = length(d);
  if (L > 1e-9) { let dl = -L / (wi + S.dcomp * S.ih2); st_add(i, wi * dl * d / L); }
}
`;

const SZ = 256; // шаг динамического смещения uniform-слотов
const WANT_STORAGE = 10;

export function gpuAvailable() { return typeof navigator !== 'undefined' && !!navigator.gpu; }

export async function createGPU() {
  if (!gpuAvailable()) throw new Error('WebGPU недоступен');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('нет адаптера WebGPU');
  if (adapter.limits.maxStorageBuffersPerShaderStage < WANT_STORAGE) throw new Error('мало storage-буферов: ' + adapter.limits.maxStorageBuffersPerShaderStage);
  const hasTS = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: WANT_STORAGE }, requiredFeatures: hasTS ? ['timestamp-query'] : [] });
  const U = GPUBufferUsage;
  const ts = hasTS ? { qs: device.createQuerySet({ type: 'timestamp', count: 2 }), res: device.createBuffer({ size: 16, usage: U.QUERY_RESOLVE | U.COPY_SRC }), rd: device.createBuffer({ size: 16, usage: U.COPY_DST | U.MAP_READ }), busy: false } : null;
  const entries = [];
  for (let i = 0; i <= 7; i++) entries.push({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: i >= 3 && i <= 6 ? 'read-only-storage' : 'storage' } });
  entries.push({ binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } });
  entries.push({ binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
  entries.push({ binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
  const bgl = device.createBindGroupLayout({ entries });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const module = device.createShaderModule({ code: WGSL });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter(m => m.type === 'error');
  if (errs.length) throw new Error('WGSL: ' + errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
  const pipe = entryPoint => device.createComputePipeline({ layout, compute: { module, entryPoint } });
  return { device, ts, bgl, adapterInfo: adapter.info || {}, pPredict: pipe('predict'), pSolve: pipe('solve'), pPost: pipe('post'), pDrag: pipe('dragk') };
}

export class Sim3DGPU extends Sim3DWasm {
  constructor(instance, gpu) {
    super(instance, { simd: true });
    this.gpu = gpu; this.statMs = 0; this.gpuMs = 0; this.stepMs = 0; this.skipped = 0; this._tLast = 0; this.wantDefo = true;
    this._gen = 0; this._inflight = 0; this._rbBusy = false; this._frame = 0; this._buf = null;
  }

  buildBox(...a) { super.buildBox(...a); this._gpuInit(); }

  _gpuInit() {
    const { device, bgl } = this.gpu, n = this.n, T = this.T, nc = this.ncolors, ord = this._ord, cst = this._cstart;
    this._gen++;
    if (this._buf) for (const k in this._buf) this._buf[k].destroy();
    const order = [], offs = [], cnts = [];
    for (let c = 0; c < nc; c++) { offs.push(order.length); for (let s = cst[c]; s < cst[c + 1]; s++) if (ord[s] >= 0) order.push(ord[s]); cnts.push(order.length - offs[c]); }
    const tets = new Uint32Array(T * 4), dm = new Float32Array(T * 9), iv = new Float32Array(T), quat = new Float32Array(T * 4);
    for (let i = 0; i < T; i++) {
      const t = order[i];
      for (let k = 0; k < 4; k++) tets[i * 4 + k] = this.tets[t * 4 + k];
      for (let k = 0; k < 9; k++) dm[i * 9 + k] = this.dmInv[t * 9 + k];
      iv[i] = 1 / this.rest[t]; quat[i * 4 + 3] = 1;
    }
    this._order = Int32Array.from(order); this._offs = offs; this._cnts = cnts;
    const f32 = a => Float32Array.from(a);
    const U = GPUBufferUsage;
    const mk = (data, usage) => { const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | U.COPY_DST | U.COPY_SRC, mappedAtCreation: true }); new data.constructor(b.getMappedRange()).set(data); b.unmap(); return b; };
    const B = this._buf = {};
    B.pos = mk(f32(this.x.subarray(0, n * 3)), U.STORAGE); B.prv = mk(f32(this.x.subarray(0, n * 3)), U.STORAGE); B.vel = mk(new Float32Array(n * 3), U.STORAGE);
    B.w = mk(f32(this.w.subarray(0, n)), U.STORAGE); B.tets = mk(tets, U.STORAGE); B.dm = mk(dm, U.STORAGE); B.iv = mk(iv, U.STORAGE); B.quat = mk(quat, U.STORAGE);
    B.defo = mk(new Float32Array(T), U.STORAGE); B.drag = mk(new Float32Array(DRAG_CAP * 4), U.STORAGE);
    B.u = device.createBuffer({ size: (3 + nc) * SZ, usage: U.UNIFORM | U.COPY_DST });
    B.sPos = device.createBuffer({ size: Math.max(16, n * 12), usage: U.COPY_DST | U.MAP_READ });
    B.sDef = device.createBuffer({ size: Math.max(16, T * 4), usage: U.COPY_DST | U.MAP_READ });
    this._ub = new ArrayBuffer((3 + nc) * SZ); this._uv = new DataView(this._ub);
    this._dragBuf = new ArrayBuffer(DRAG_CAP * 16); this._dragF = new Float32Array(this._dragBuf); this._dragU = new Uint32Array(this._dragBuf);
    const res = ['pos', 'prv', 'vel', 'w', 'tets', 'dm', 'iv', 'quat'].map((k, i) => ({ binding: i, resource: { buffer: B[k] } }));
    res.push({ binding: 8, resource: { buffer: B.u, size: SZ } }, { binding: 9, resource: { buffer: B.defo } }, { binding: 10, resource: { buffer: B.drag } });
    this._bg = device.createBindGroup({ layout: bgl, entries: res });
    this._rbBusy = false; this._inflight = 0;
  }

  pinMinX(on, eps) {
    super.pinMinX(on, eps);
    if (this._buf) this.gpu.device.queue.writeBuffer(this._buf.w, 0, Float32Array.from(this.w.subarray(0, this.n)));
  }

  step(dt) {
    if (!this.bound || !this._buf) return false;
    if (this._inflight >= 2) { this.skipped++; return false; } // GPU не успевает — кадр физики пропускаем, очередь не копим
    const { device, pPredict, pSolve, pPost, pDrag } = this.gpu, B = this._buf, n = this.n, nc = this.ncolors, S = this.substeps, h = dt / S, d = this.drag, dn = d.n;
    const { mu, lam } = this._lame(), uv = this._uv;
    const put = (i, count, off) => {
      const o = i * SZ; uv.setUint32(o, count, true); uv.setUint32(o + 4, off, true); uv.setUint32(o + 8, n, true);
      [h, 1 / (h * h), mu, lam, this.gravity, this.damping, this.friction, this.floorY, d.compliance].forEach((v, k) => uv.setFloat32(o + 16 + k * 4, v, true));
    };
    put(0, n, 0); put(1, n, 0); put(2, dn, 0);
    for (let c = 0; c < nc; c++) put(3 + c, this._cnts[c], this._offs[c]);
    device.queue.writeBuffer(B.u, 0, this._ub);
    if (dn > 0) {
      const f = this._dragF, u = this._dragU;
      for (let k = 0; k < dn; k++) { f[k * 4] = d.tx + d.off[k * 3]; f[k * 4 + 1] = d.ty + d.off[k * 3 + 1]; f[k * 4 + 2] = d.tz + d.off[k * 3 + 2]; u[k * 4 + 3] = d.list[k]; }
      device.queue.writeBuffer(B.drag, 0, this._dragBuf, 0, dn * 16);
    }
    const ts = this.gpu.ts, useTS = ts && !ts.busy && (this._frame & 7) === 0, now = performance.now();
    if (this._tLast) this.stepMs = this.stepMs ? this.stepMs * 0.9 + (now - this._tLast) * 0.1 : now - this._tLast;
    this._tLast = now;
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass(useTS ? { timestampWrites: { querySet: ts.qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined), bg = this._bg;
    for (let s = 0; s < S; s++) {
      pass.setPipeline(pPredict); pass.setBindGroup(0, bg, [0]); pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.setPipeline(pSolve);
      for (let c = 0; c < nc; c++) { pass.setBindGroup(0, bg, [(3 + c) * SZ]); pass.dispatchWorkgroups(Math.ceil(this._cnts[c] / 64)); }
      if (dn > 0) { pass.setPipeline(pDrag); pass.setBindGroup(0, bg, [2 * SZ]); pass.dispatchWorkgroups(Math.ceil(dn / 64)); }
      pass.setPipeline(pPost); pass.setBindGroup(0, bg, [SZ]); pass.dispatchWorkgroups(Math.ceil(n / 64));
    }
    pass.end();
    if (useTS) { enc.resolveQuerySet(ts.qs, 0, 2, ts.res, 0); enc.copyBufferToBuffer(ts.res, 0, ts.rd, 0, 16); }
    this._frame++;
    const rb = !this._rbBusy, rbDef = rb && this.wantDefo && (this._frame & 3) === 1;
    if (rb) { enc.copyBufferToBuffer(B.pos, 0, B.sPos, 0, n * 12); if (rbDef) enc.copyBufferToBuffer(B.defo, 0, B.sDef, 0, this.T * 4); }
    device.queue.submit([enc.finish()]);
    if (useTS) { ts.busy = true; ts.rd.mapAsync(GPUMapMode.READ).then(() => { const t = new BigUint64Array(ts.rd.getMappedRange().slice(0)); ts.rd.unmap(); const g = Number(t[1] - t[0]) / 1e6; this.gpuMs = this.gpuMs ? this.gpuMs * 0.7 + g * 0.3 : g; ts.busy = false; }).catch(() => { ts.busy = false; }); }
    const t0 = performance.now(), gen = this._gen; this._inflight++;
    device.queue.onSubmittedWorkDone().then(() => { this._inflight--; this.statMs = this.statMs * 0.9 + (performance.now() - t0) * 0.1; });
    if (rb) {
      this._rbBusy = true;
      const pr = [B.sPos.mapAsync(GPUMapMode.READ)]; if (rbDef) pr.push(B.sDef.mapAsync(GPUMapMode.READ));
      Promise.all(pr).then(() => {
        if (gen !== this._gen) return;
        const a = new Float32Array(B.sPos.getMappedRange()), x = this.x; for (let i = 0, m = n * 3; i < m; i++) x[i] = a[i];
        B.sPos.unmap();
        if (rbDef) { const q = new Float32Array(B.sDef.getMappedRange()), o = this._order, df = this.defo; for (let i = 0; i < o.length; i++) df[o[i]] = q[i]; B.sDef.unmap(); }
        this._rbBusy = false;
      }).catch(() => { if (gen === this._gen) this._rbBusy = false; });
    }
    return true;
  }
}
