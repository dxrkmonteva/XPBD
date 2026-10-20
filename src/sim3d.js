// M8: 3D мягкое тело из тетраэдров. XPBD (малые шаги, λ = 0 в начале подшага) + корротационный FEM.
// Те же идеи, что в 2D (sim.js), но F — матрица 3×3, а ограничения:
//   девиаторное  C = ||F − R||_F,  α = 1/(2 μ V0),  ∂C/∂F = (F − R)/C
//   объёмное     C = det F − 1,    α = 1/(λ V0),    ∂C/∂F = cof(F)   (устойчиво и при вывернутом элементе)
// R — ближайший поворот к F: извлекается итерациями по кватерниону (Müller et al. 2016) с «тёплым стартом» от
// прошлого подшага, поэтому хватает одной-двух итераций на элемент.
// Единицы СИ: метры, килограммы, секунды; плотность ρ = 1000 кг/м³ (масса вершины = ρ · треть/четверть объёма соседей).

export const DRAG_CAP = 4096; // макс. число захваченных вершин

export class Sim3D {
  constructor() {
    this.gravity = 9.81;
    this.young = 5e5;          // модуль Юнга E, Па
    this.nu = 0.3;             // коэффициент Пуассона
    this.density = 1000;
    this.damping = 0.999;
    this.friction = 0.6;       // трение об пол (доля касательного смещения, которая гасится)
    this.substeps = 10;
    this.floorY = 0;
    this.n = 0; this.T = 0;
    this.drag = { n: 0, list: new Int32Array(DRAG_CAP), off: new Float64Array(DRAG_CAP * 3), tx: 0, ty: 0, tz: 0, compliance: 1e-4 };
    this.clear();
  }

  clear() {
    this.n = 0; this.T = 0;
    this.x = new Float64Array(0); this.p = new Float64Array(0); this.v = new Float64Array(0); this.w = new Float64Array(0);
    this.tets = new Int32Array(0); this.dmInv = new Float64Array(0); this.vol = new Float64Array(0); this.rest = new Float64Array(0);
    this.quat = new Float64Array(0); this.faces = new Int32Array(0); this.defo = new Float32Array(0);
    this.drag.n = 0; this._grabs = null; this._w0 = null;
  }

  // Прямоугольный брус nx×ny×nz кубиков со стороной h, левый-нижний-ближний угол в (ox, oy, oz).
  // Каждый кубик → 6 тетраэдров (разбиение Куна — согласовано между соседними кубиками).
  buildBox(nx, ny, nz, h, ox, oy, oz) {
    this.clear();
    const cx = nx + 1, cy = ny + 1, cz = nz + 1, n = cx * cy * cz;
    this.n = n;
    this.x = new Float64Array(n * 3); this.p = new Float64Array(n * 3); this.v = new Float64Array(n * 3); this.w = new Float64Array(n).fill(1);
    const id = (i, j, k) => (k * cy + j) * cx + i;
    for (let k = 0; k < cz; k++) for (let j = 0; j < cy; j++) for (let i = 0; i < cx; i++) {
      const a = id(i, j, k) * 3;
      this.x[a] = this.p[a] = ox + i * h; this.x[a + 1] = this.p[a + 1] = oy + j * h; this.x[a + 2] = this.p[a + 2] = oz + k * h;
    }
    const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    const tets = [];
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      for (const pm of perms) {
        const c = [i, j, k], vs = [id(c[0], c[1], c[2])];
        for (let s = 0; s < 3; s++) { c[pm[s]]++; vs.push(id(c[0], c[1], c[2])); }
        tets.push(vs[0], vs[1], vs[2], vs[3]);
      }
    }
    this.T = tets.length / 4;
    this.tets = Int32Array.from(tets);
    this._prepare();
    this.spacing = h;
  }

  // Dm⁻¹, объёмы, массы, поверхность. Вызывается после построения сетки.
  _prepare() {
    const T = this.T, x = this.x, tets = this.tets;
    this.dmInv = new Float64Array(T * 9); this.vol = new Float64Array(T); this.quat = new Float64Array(T * 4);
    this.defo = new Float32Array(T);
    const m = new Float64Array(9);
    for (let t = 0; t < T; t++) {
      let a = tets[t * 4] * 3, b = tets[t * 4 + 1] * 3, c = tets[t * 4 + 2] * 3, d = tets[t * 4 + 3] * 3;
      // столбцы Dm: x1−x0, x2−x0, x3−x0
      let det = this._dm(a, b, c, d, m);
      if (det < 0) { const s = tets[t * 4 + 1]; tets[t * 4 + 1] = tets[t * 4 + 2]; tets[t * 4 + 2] = s; b = tets[t * 4 + 1] * 3; c = tets[t * 4 + 2] * 3; det = this._dm(a, b, c, d, m); }
      this.vol[t] = det / 6;
      this._inv3(m, this.dmInv, t * 9);
      this.quat[t * 4 + 3] = 1;
    }
    // массы вершин
    const vv = new Float64Array(this.n);
    for (let t = 0; t < T; t++) for (let k = 0; k < 4; k++) vv[tets[t * 4 + k]] += this.vol[t] / 4;
    this.w = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) this.w[i] = vv[i] > 0 ? 1 / (this.density * vv[i]) : 0;
    this.rest = Float64Array.from(this.vol);
    // поверхность: грани, встречающиеся один раз; ориентация — наружу (от противолежащей вершины)
    const map = new Map(), faceOf = [[1, 2, 3, 0], [0, 3, 2, 1], [0, 1, 3, 2], [0, 2, 1, 3]];
    const N = this.n;
    for (let t = 0; t < T; t++) for (const [i, j, k, o] of faceOf) {
      const A = tets[t * 4 + i], B = tets[t * 4 + j], C = tets[t * 4 + k], O = tets[t * 4 + o];
      const s = [A, B, C].sort((p, q) => p - q), key = (s[0] * N + s[1]) * N + s[2];
      if (map.has(key)) map.get(key).cnt++; else map.set(key, { A, B, C, O, cnt: 1 });
    }
    const f = [];
    for (const e of map.values()) {
      if (e.cnt !== 1) continue;
      const pa = e.A * 3, pb = e.B * 3, pc = e.C * 3, po = e.O * 3;
      const ux = x[pb] - x[pa], uy = x[pb + 1] - x[pa + 1], uz = x[pb + 2] - x[pa + 2];
      const vx = x[pc] - x[pa], vy = x[pc + 1] - x[pa + 1], vz = x[pc + 2] - x[pa + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const dot = nx * (x[po] - x[pa]) + ny * (x[po + 1] - x[pa + 1]) + nz * (x[po + 2] - x[pa + 2]);
      if (dot > 0) f.push(e.A, e.C, e.B); else f.push(e.A, e.B, e.C); // нормаль смотрит от противолежащей вершины
    }
    this.faces = Int32Array.from(f);
  }

  _dm(a, b, c, d, m) {
    const x = this.x;
    m[0] = x[b] - x[a]; m[3] = x[b + 1] - x[a + 1]; m[6] = x[b + 2] - x[a + 2];
    m[1] = x[c] - x[a]; m[4] = x[c + 1] - x[a + 1]; m[7] = x[c + 2] - x[a + 2];
    m[2] = x[d] - x[a]; m[5] = x[d + 1] - x[a + 1]; m[8] = x[d + 2] - x[a + 2];
    return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
  }

  _inv3(m, out, o) {
    const a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
    const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g, det = a * A + b * B + c * C, r = 1 / det;
    out[o] = A * r; out[o + 1] = (c * h - b * i) * r; out[o + 2] = (b * f - c * e) * r;
    out[o + 3] = B * r; out[o + 4] = (a * i - c * g) * r; out[o + 5] = (c * d - a * f) * r;
    out[o + 6] = C * r; out[o + 7] = (b * g - a * h) * r; out[o + 8] = (a * e - b * d) * r;
  }

  _lame() { const E = this.young, nu = this.nu; return { mu: E / (2 * (1 + nu)), lam: E * nu / ((1 + nu) * (1 - 2 * nu)) }; }

  // F = Ds · Dm⁻¹ → F (row-major 9) ; возвращает без аллокаций в F
  _defGrad(t, F) {
    const x = this.x, tets = this.tets, M = this.dmInv, o = t * 9;
    const a = tets[t * 4] * 3, b = tets[t * 4 + 1] * 3, c = tets[t * 4 + 2] * 3, d = tets[t * 4 + 3] * 3;
    const d1x = x[b] - x[a], d1y = x[b + 1] - x[a + 1], d1z = x[b + 2] - x[a + 2];
    const d2x = x[c] - x[a], d2y = x[c + 1] - x[a + 1], d2z = x[c + 2] - x[a + 2];
    const d3x = x[d] - x[a], d3y = x[d + 1] - x[a + 1], d3z = x[d + 2] - x[a + 2];
    // F[r][c] = Σ_k Ds[r][k] · Dminv[k][c]
    F[0] = d1x * M[o] + d2x * M[o + 3] + d3x * M[o + 6]; F[1] = d1x * M[o + 1] + d2x * M[o + 4] + d3x * M[o + 7]; F[2] = d1x * M[o + 2] + d2x * M[o + 5] + d3x * M[o + 8];
    F[3] = d1y * M[o] + d2y * M[o + 3] + d3y * M[o + 6]; F[4] = d1y * M[o + 1] + d2y * M[o + 4] + d3y * M[o + 7]; F[5] = d1y * M[o + 2] + d2y * M[o + 5] + d3y * M[o + 8];
    F[6] = d1z * M[o] + d2z * M[o + 3] + d3z * M[o + 6]; F[7] = d1z * M[o + 1] + d2z * M[o + 4] + d3z * M[o + 7]; F[8] = d1z * M[o + 2] + d2z * M[o + 5] + d3z * M[o + 8];
  }

  // Ближайший поворот к F (кватернион q хранится по элементу, уточняется iters итерациями)
  _rotation(t, F, R, iters) {
    const q = this.quat, o = t * 4;
    let qx = q[o], qy = q[o + 1], qz = q[o + 2], qw = q[o + 3];
    for (let it = 0; it < iters; it++) {
      // R из кватерниона (row-major)
      const xx = qx * qx, yy = qy * qy, zz = qz * qz, xy = qx * qy, xz = qx * qz, yz = qy * qz, wx = qw * qx, wy = qw * qy, wz = qw * qz;
      R[0] = 1 - 2 * (yy + zz); R[1] = 2 * (xy - wz); R[2] = 2 * (xz + wy);
      R[3] = 2 * (xy + wz); R[4] = 1 - 2 * (xx + zz); R[5] = 2 * (yz - wx);
      R[6] = 2 * (xz - wy); R[7] = 2 * (yz + wx); R[8] = 1 - 2 * (xx + yy);
      // ω = Σ r_i × f_i / (|Σ r_i·f_i| + ε)   (i — столбцы)
      let ox = 0, oy = 0, oz = 0, dot = 0;
      for (let c = 0; c < 3; c++) {
        const rx = R[c], ry = R[3 + c], rz = R[6 + c], fx = F[c], fy = F[3 + c], fz = F[6 + c];
        ox += ry * fz - rz * fy; oy += rz * fx - rx * fz; oz += rx * fy - ry * fx; dot += rx * fx + ry * fy + rz * fz;
      }
      const k = 1 / (Math.abs(dot) + 1e-9); ox *= k; oy *= k; oz *= k;
      if (ox * ox + oy * oy + oz * oz < 1e-18) break;
      // dq = (ω/2, 1) без нормировки (≈ поворот на угол |ω|): без sin/cos — одинаковые числа в JS и WASM
      const sx = ox * 0.5, sy = oy * 0.5, sz = oz * 0.5;
      // q ← dq ⊗ q
      const nx = qx + sx * qw + sy * qz - sz * qy, ny = qy + sy * qw + sz * qx - sx * qz;
      const nz = qz + sz * qw + sx * qy - sy * qx, nw = qw - sx * qx - sy * qy - sz * qz;
      const l = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
      qx = nx * l; qy = ny * l; qz = nz * l; qw = nw * l;
    }
    q[o] = qx; q[o + 1] = qy; q[o + 2] = qz; q[o + 3] = qw;
    const xx = qx * qx, yy = qy * qy, zz = qz * qz, xy = qx * qy, xz = qx * qz, yz = qy * qz, wx = qw * qx, wy = qw * qy, wz = qw * qz;
    R[0] = 1 - 2 * (yy + zz); R[1] = 2 * (xy - wz); R[2] = 2 * (xz + wy);
    R[3] = 2 * (xy + wz); R[4] = 1 - 2 * (xx + zz); R[5] = 2 * (yz - wx);
    R[6] = 2 * (xz - wy); R[7] = 2 * (yz + wx); R[8] = 1 - 2 * (xx + yy);
  }

  // XPBD-коррекция для ограничения C(F) с градиентом G = ∂C/∂F: H = G · Dm⁻ᵀ, ∇x_{c+1} = H[:,c], ∇x0 = −Σ
  _apply(t, G, C, alphaBar) {
    const x = this.x, w = this.w, tets = this.tets, M = this.dmInv, o = t * 9;
    const H = this._H;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[r * 3 + c] = G[r * 3] * M[o + c * 3] + G[r * 3 + 1] * M[o + c * 3 + 1] + G[r * 3 + 2] * M[o + c * 3 + 2];
    const h0x = -(H[0] + H[1] + H[2]), h0y = -(H[3] + H[4] + H[5]), h0z = -(H[6] + H[7] + H[8]);
    const p0 = tets[t * 4], p1 = tets[t * 4 + 1], p2 = tets[t * 4 + 2], p3 = tets[t * 4 + 3];
    const w0 = w[p0], w1 = w[p1], w2 = w[p2], w3 = w[p3];
    const den = w0 * (h0x * h0x + h0y * h0y + h0z * h0z) + w1 * (H[0] * H[0] + H[3] * H[3] + H[6] * H[6]) +
      w2 * (H[1] * H[1] + H[4] * H[4] + H[7] * H[7]) + w3 * (H[2] * H[2] + H[5] * H[5] + H[8] * H[8]) + alphaBar;
    if (den < 1e-30) return;
    const dl = -C / den;
    x[p0 * 3] += w0 * dl * h0x; x[p0 * 3 + 1] += w0 * dl * h0y; x[p0 * 3 + 2] += w0 * dl * h0z;
    x[p1 * 3] += w1 * dl * H[0]; x[p1 * 3 + 1] += w1 * dl * H[3]; x[p1 * 3 + 2] += w1 * dl * H[6];
    x[p2 * 3] += w2 * dl * H[1]; x[p2 * 3 + 1] += w2 * dl * H[4]; x[p2 * 3 + 2] += w2 * dl * H[7];
    x[p3 * 3] += w3 * dl * H[2]; x[p3 * 3 + 1] += w3 * dl * H[5]; x[p3 * 3 + 2] += w3 * dl * H[8];
  }

  femSolve(h) {
    const T = this.T, vol = this.rest, { mu, lam } = this._lame(), h2 = h * h;
    const F = this._F || (this._F = new Float64Array(9)), R = this._R || (this._R = new Float64Array(9)), G = this._G || (this._G = new Float64Array(9));
    this._H = this._H || new Float64Array(9);
    for (let t = 0; t < T; t++) {
      // 1) девиаторная часть
      this._defGrad(t, F);
      this._rotation(t, F, R, 2);
      let s = 0;
      for (let i = 0; i < 9; i++) { const e = F[i] - R[i]; G[i] = e; s += e * e; }
      const C = Math.sqrt(s);
      this.defo[t] = C; // мера деформации для раскраски (последний подшаг)
      if (C > 1e-9) { for (let i = 0; i < 9; i++) G[i] /= C; this._apply(t, G, C, 1 / (2 * mu * vol[t]) / h2); }
      // 2) объёмная часть: C = det F − 1
      this._defGrad(t, F);
      const f0x = F[0], f0y = F[3], f0z = F[6], f1x = F[1], f1y = F[4], f1z = F[7], f2x = F[2], f2y = F[5], f2z = F[8];
      const det = f0x * (f1y * f2z - f1z * f2y) - f1x * (f0y * f2z - f0z * f2y) + f2x * (f0y * f1z - f0z * f1y);
      if (lam > 0 && Math.abs(det - 1) > 1e-12) {
        // столбцы градиента: ∂det/∂f0 = f1×f2, ∂/∂f1 = f2×f0, ∂/∂f2 = f0×f1
        G[0] = f1y * f2z - f1z * f2y; G[3] = f1z * f2x - f1x * f2z; G[6] = f1x * f2y - f1y * f2x;
        G[1] = f2y * f0z - f2z * f0y; G[4] = f2z * f0x - f2x * f0z; G[7] = f2x * f0y - f2y * f0x;
        G[2] = f0y * f1z - f0z * f1y; G[5] = f0z * f1x - f0x * f1z; G[8] = f0x * f1y - f0y * f1x;
        this._apply(t, G, det - 1, 1 / (lam * vol[t]) / h2);
      }
    }
  }

  // Захват области: все вершины в радиусе radius от точки (px, py, pz) тянутся мягкими пружинами к цели
  // (tx, ty, tz) + их смещение относительно точки (область движется как целое — нет «шипа» на мелкой сетке).
  // Захватов может быть несколько одновременно (по одному на палец/указатель): id — любой ключ.
  // В drag.list/off лежат абсолютные цели каждой вершины (drag.tx=ty=tz=0), ядро (JS/WASM) от числа захватов не зависит.
  startGrab(px, py, pz, radius, tx, ty, tz, id = 0) {
    const x = this.x, w = this.w, r2 = radius * radius, idx = [], off = [];
    this._grabs = this._grabs || new Map();
    this._grabs.delete(id);
    let used = 0; for (const g of this._grabs.values()) used += g.idx.length;
    for (let i = 0; i < this.n && used + idx.length < DRAG_CAP; i++) {
      const dx = x[i * 3] - px, dy = x[i * 3 + 1] - py, dz = x[i * 3 + 2] - pz;
      if (dx * dx + dy * dy + dz * dz <= r2 && w[i] > 0) { idx.push(i); off.push(dx, dy, dz); }
    }
    this._grabs.set(id, { idx, off, t: [tx, ty, tz] });
    this._packGrabs();
    return idx.length;
  }
  moveGrab(tx, ty, tz, id = 0) {
    const g = this._grabs && this._grabs.get(id); if (!g) return;
    g.t[0] = tx; g.t[1] = ty; g.t[2] = tz; this._packGrabs();
  }
  release(id) {
    if (!this._grabs) { this.drag.n = 0; return; }
    if (id === undefined) this._grabs.clear(); else this._grabs.delete(id);
    this._packGrabs();
  }
  get grabCount() { return this._grabs ? this._grabs.size : 0; }
  _packGrabs() {
    const d = this.drag; let n = 0;
    for (const g of this._grabs.values()) {
      for (let k = 0; k < g.idx.length && n < DRAG_CAP; k++, n++) {
        d.list[n] = g.idx[k];
        d.off[n * 3] = g.t[0] + g.off[k * 3]; d.off[n * 3 + 1] = g.t[1] + g.off[k * 3 + 1]; d.off[n * 3 + 2] = g.t[2] + g.off[k * 3 + 2];
      }
    }
    d.n = n; d.tx = d.ty = d.tz = 0;
  }

  // Закрепить вершины с координатой x в пределах eps от минимума (левая грань бруса)
  pinMinX(on, eps = 1e-6) {
    let mn = Infinity;
    for (let i = 0; i < this.n; i++) mn = Math.min(mn, this.x[i * 3]);
    if (!this._w0) this._w0 = Float64Array.from(this.w);
    for (let i = 0; i < this.n; i++) if (this.x[i * 3] < mn + eps) this.w[i] = on ? 0 : this._w0[i];
  }

  step(dt) {
    const { n, x, p, v, w } = this, S = this.substeps, h = dt / S, T = this.T;
    for (let s = 0; s < S; s++) {
      for (let i = 0; i < n; i++) {
        const a = i * 3;
        p[a] = x[a]; p[a + 1] = x[a + 1]; p[a + 2] = x[a + 2];
        if (w[i] > 0) { v[a + 1] -= this.gravity * h; x[a] += v[a] * h; x[a + 1] += v[a + 1] * h; x[a + 2] += v[a + 2] * h; }
      }
      this.femSolve(h);
      const d = this.drag;
      for (let k = 0; k < d.n; k++) { // мягкие пружины к цели: C = |x − (цель + смещение)|
        const i = d.list[k], wi = w[i]; if (wi <= 0) continue;
        const a = i * 3, dx = x[a] - (d.tx + d.off[k * 3]), dy = x[a + 1] - (d.ty + d.off[k * 3 + 1]), dz = x[a + 2] - (d.tz + d.off[k * 3 + 2]), L = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (L > 1e-9) { const dl = -L / (wi + d.compliance / (h * h)); x[a] += wi * dl * dx / L; x[a + 1] += wi * dl * dy / L; x[a + 2] += wi * dl * dz / L; }
      }
      const fy = this.floorY, fr = 1 - this.friction;
      for (let i = 0; i < n; i++) {
        if (w[i] === 0) continue;
        const a = i * 3;
        if (x[a + 1] < fy) { x[a + 1] = fy; x[a] = p[a] + (x[a] - p[a]) * fr; x[a + 2] = p[a + 2] + (x[a + 2] - p[a + 2]) * fr; }
      }
      const k = this.damping / h;
      for (let i = 0; i < n * 3; i++) v[i] = (x[i] - p[i]) * k;
      for (let i = 0; i < n; i++) if (w[i] === 0) v[i * 3] = v[i * 3 + 1] = v[i * 3 + 2] = 0;
    }
  }

  // Диагностика
  invertedCount() {
    const x = this.x, tets = this.tets, m = new Float64Array(9); let k = 0;
    for (let t = 0; t < this.T; t++) {
      const det = this._dm(tets[t * 4] * 3, tets[t * 4 + 1] * 3, tets[t * 4 + 2] * 3, tets[t * 4 + 3] * 3, m);
      if (det / 6 * this.rest[t] <= 0) k++;
    }
    return k;
  }
  volumeRatio() {
    const m = new Float64Array(9), tets = this.tets; let cur = 0, r0 = 0;
    for (let t = 0; t < this.T; t++) { cur += this._dm(tets[t * 4] * 3, tets[t * 4 + 1] * 3, tets[t * 4 + 2] * 3, tets[t * 4 + 3] * 3, m) / 6; r0 += this.rest[t]; }
    return cur / r0;
  }
  kinetic() { let e = 0; for (let i = 0; i < this.n; i++) if (this.w[i] > 0) e += 0.5 / this.w[i] * (this.v[i * 3] ** 2 + this.v[i * 3 + 1] ** 2 + this.v[i * 3 + 2] ** 2); return e; }
}
