// GPU-1: проверка WebGPU и замер: тот же тетраэдральный шаг XPBD-FEM (раскраска, 10 подшагов), что в wasm/core3d.cpp,
// но в WGSL compute-шейдерах. Сравнивается с CPU (WASM SIMD) на этом же устройстве. Физику GPU здесь не выводим — только время.
import { Sim3DWasm, loadCore3D } from './wasm-sim3d.js';
import simdUrl from './wasm/core3d.simd.wasm?url';

const out = document.getElementById('out');
const log = s => { out.textContent += '\n' + s; };

const WGSL = /* wgsl */`
struct Slot { count: u32, off: u32, n: u32, pad: u32, h: f32, ih2: f32, mu: f32, lam: f32, grav: f32, damp: f32, fric: f32, floorY: f32 };
@group(0) @binding(0) var<storage, read_write> pos: array<f32>;
@group(0) @binding(1) var<storage, read_write> prv: array<f32>;
@group(0) @binding(2) var<storage, read_write> vel: array<f32>;
@group(0) @binding(3) var<storage, read> wgt: array<f32>;
@group(0) @binding(4) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> dm: array<f32>;
@group(0) @binding(6) var<storage, read> invV: array<f32>;
@group(0) @binding(7) var<storage, read_write> quat: array<vec4<f32>>;
@group(0) @binding(8) var<uniform> S: Slot;

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
`;

async function gpuBench(sim, frames, substeps) {
  const adapter = await navigator.gpu.requestAdapter();
  const hasTS = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(8, adapter.limits.maxStorageBuffersPerShaderStage) }, requiredFeatures: hasTS ? ['timestamp-query'] : [] });
  const n = sim.n, T = sim.T, nc = sim.ncolors, ord = sim._ord, cst = sim._cstart;
  // компактный порядок тетраэдров по цветам (без пустых слотов)
  const order = [], offs = [], cnts = [];
  for (let c = 0; c < nc; c++) { offs.push(order.length); for (let s = cst[c]; s < cst[c + 1]; s++) if (ord[s] >= 0) order.push(ord[s]); cnts.push(order.length - offs[c]); }
  const tets = new Uint32Array(T * 4), dm = new Float32Array(T * 9), iv = new Float32Array(T), quat = new Float32Array(T * 4);
  for (let i = 0; i < T; i++) {
    const t = order[i];
    for (let k = 0; k < 4; k++) tets[i * 4 + k] = sim.tets[t * 4 + k];
    for (let k = 0; k < 9; k++) dm[i * 9 + k] = sim.dmInv[t * 9 + k];
    iv[i] = 1 / sim.rest[t]; quat[i * 4 + 3] = 1;
  }
  const f32 = a => Float32Array.from(a);
  const mk = (data, usage) => { const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, mappedAtCreation: true }); new data.constructor(b.getMappedRange()).set(data); b.unmap(); return b; };
  const ST = GPUBufferUsage.STORAGE;
  const bPos = mk(f32(sim.x.subarray(0, n * 3)), ST), bPrv = mk(f32(sim.x.subarray(0, n * 3)), ST), bVel = mk(new Float32Array(n * 3), ST);
  const bW = mk(f32(sim.w.subarray(0, n)), ST), bT = mk(tets, ST), bDm = mk(dm, ST), bIv = mk(iv, ST), bQ = mk(quat, ST);
  // uniform-слоты (по 256 байт): predict, post, цвета…
  const slots = 2 + nc, SZ = 256, ub = new ArrayBuffer(slots * SZ), uv = new DataView(ub);
  const E = 1e6 * 0.5, nu = 0.3, mu = E / (2 * (1 + nu)), lam = E * nu / ((1 + nu) * (1 - 2 * nu)), h = (1 / 60) / substeps;
  const put = (i, count, off) => {
    const o = i * SZ; uv.setUint32(o, count, true); uv.setUint32(o + 4, off, true); uv.setUint32(o + 8, n, true);
    [h, 1 / (h * h), mu, lam, 9.81, 0.999, 0.6, 0].forEach((v, k) => uv.setFloat32(o + 16 + k * 4, v, true));
  };
  put(0, n, 0); put(1, n, 0); for (let c = 0; c < nc; c++) put(2 + c, cnts[c], offs[c]);
  const bU = device.createBuffer({ size: ub.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(bU, 0, ub);
  const entries = [0, 1, 2, 3, 4, 5, 6, 7].map(i => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: i === 3 || i === 4 || i === 5 || i === 6 ? 'read-only-storage' : 'storage' } }));
  entries.push({ binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } });
  const bgl = device.createBindGroupLayout({ entries });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const module = device.createShaderModule({ code: WGSL });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter(m => m.type === 'error');
  if (errs.length) throw new Error('WGSL: ' + errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
  const pipe = name => device.createComputePipeline({ layout, compute: { module, entryPoint: name } });
  const pPredict = pipe('predict'), pPost = pipe('post'), pSolve = pipe('solve');
  const bg = device.createBindGroup({ layout: bgl, entries: [bPos, bPrv, bVel, bW, bT, bDm, bIv, bQ].map((b, i) => ({ binding: i, resource: { buffer: b } })).concat([{ binding: 8, resource: { buffer: bU, size: 256 } }]) });
  const qs = hasTS ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const qRes = hasTS ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const qRead = hasTS ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
  const frame = (timed = false) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(timed && qs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
    for (let s = 0; s < substeps; s++) {
      pass.setPipeline(pPredict); pass.setBindGroup(0, bg, [0]); pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.setPipeline(pSolve);
      for (let c = 0; c < nc; c++) { pass.setBindGroup(0, bg, [(2 + c) * SZ]); pass.dispatchWorkgroups(Math.ceil(cnts[c] / 64)); }
      pass.setPipeline(pPost); pass.setBindGroup(0, bg, [SZ]); pass.dispatchWorkgroups(Math.ceil(n / 64));
    }
    pass.end();
    if (timed && qs) { enc.resolveQuerySet(qs, 0, 2, qRes, 0); enc.copyBufferToBuffer(qRes, 0, qRead, 0, 16); }
    device.queue.submit([enc.finish()]);
  };
  for (let i = 0; i < 3; i++) frame();
  await device.queue.onSubmittedWorkDone();
  // 1) задержка: кадр → ожидание завершения (то, что видит один кадр)
  const times = [];
  for (let i = 0; i < frames; i++) { const t0 = performance.now(); frame(); await device.queue.onSubmittedWorkDone(); times.push(performance.now() - t0); }
  times.sort((a, b) => a - b);
  // 2) пропускная способность: кадры подряд без ожидания (как при реальной игре, когда CPU занят чем-то другим)
  const tp0 = performance.now(); for (let i = 0; i < frames; i++) frame(); await device.queue.onSubmittedWorkDone();
  const throughput = (performance.now() - tp0) / frames;
  // 3) чистое время работы GPU по аппаратным таймерам (без накладных расходов CPU/драйвера на отправку)
  let gpuMs = null;
  if (hasTS) {
    const g = [];
    for (let i = 0; i < 6; i++) { frame(true); await qRead.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qRead.getMappedRange().slice(0)); qRead.unmap(); g.push(Number(t[1] - t[0]) / 1e6); }
    g.sort((a, b) => a - b); gpuMs = g[g.length >> 1];
  }
  // проверка: позиции конечны, тело не улетело
  const rb = device.createBuffer({ size: n * 12, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(bPos, 0, rb, 0, n * 12); device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ); const res = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap();
  let ok = true, ymin = 1e9, ymax = -1e9; for (let i = 0; i < n; i++) { const y = res[i * 3 + 1]; if (!Number.isFinite(y)) ok = false; ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); }
  device.destroy();
  return { median: times[times.length >> 1], min: times[0], throughput, gpuMs, dispatches: substeps * (nc + 2), ok, ymin, ymax, adapter };
}

async function main() {
  out.textContent = 'XPBD — проверка WebGPU\n' + navigator.userAgent;
  log('crossOriginIsolated: ' + (typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : 'n/a') + ' · secure: ' + window.isSecureContext);
  if (!navigator.gpu) { log('\n✗ navigator.gpu нет: WebGPU недоступен в этом браузере/режиме (нужен https и Chrome 121+ на Android 12+, либо включить флаг chrome://flags/#enable-unsafe-webgpu).'); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { log('\n✗ WebGPU есть, но адаптер не выдан (нет поддерживаемого GPU/драйвера).'); return; }
  const inf = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  log(`\n✓ WebGPU доступен\nАдаптер: ${inf.vendor || '?'} ${inf.architecture || ''} ${inf.device || ''} ${inf.description || ''}`.trim());
  const L = adapter.limits;
  log(`лимиты: storage-буферов на стадию ${L.maxStorageBuffersPerShaderStage}, размер storage-буфера ${(L.maxStorageBufferBindingSize / 1048576) | 0} МБ, workgroup ${L.maxComputeWorkgroupSizeX}, invocations ${L.maxComputeInvocationsPerWorkgroup}`);
  log('функции: ' + ([...adapter.features].join(', ') || '—'));
  const core = await loadCore3D(simdUrl);
  for (const k of [4, 6, 8, 10, 12]) {
    const sim = new Sim3DWasm(core, { simd: true }); sim.buildBox(8 * k, 2 * k, 2 * k, 0.25 / k, 0, 0.6, 0);
    for (let i = 0; i < 2; i++) sim.step(1 / 60);
    const N = k >= 10 ? 3 : k >= 6 ? 8 : 20, t0 = performance.now(); for (let i = 0; i < N; i++) sim.step(1 / 60);
    const cpu = (performance.now() - t0) / N;
    try {
      const g = await gpuBench(sim, k >= 8 ? 8 : 20, 10);
      log(`\n×${k}: ${sim.T} тетраэдров, ${sim.ncolors} цветов, ${g.dispatches} dispatch/кадр${g.ok ? '' : '  ⚠ NaN'}\n   CPU WASM SIMD:        ${cpu.toFixed(1)} мс/кадр\n   GPU, с ожиданием:     ${g.median.toFixed(1)} мс (лучший ${g.min.toFixed(1)})\n   GPU, конвейер:        ${g.throughput.toFixed(1)} мс/кадр   → выигрыш ×${(cpu / g.throughput).toFixed(2)}\n   GPU, чистое время:    ${g.gpuMs === null ? 'нет таймеров' : g.gpuMs.toFixed(1) + ' мс (по аппаратному таймеру)'}`);
    } catch (e) { log(`\n×${k}: GPU-тест упал: ${e.message || e}`); }
  }
  log('\nГотово. Пришли скриншот этой страницы.');
}
const run = () => main().catch(e => log('\nОшибка: ' + (e.stack || e)));
document.getElementById('run').addEventListener('click', run);
run();
