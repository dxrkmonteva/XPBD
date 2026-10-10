// Ядро симуляции: 2D XPBD с «малыми шагами» (small steps).
// Позже этот файл переписывается на C++ (тот же интерфейс → WASM).

export class Sim {
  constructor() {
    this.gravity = 900;       // px/s^2
    this.compliance = 1e-5;   // обратная жёсткость рёбер (0 = абсолютно жёсткий)
    this.areaEnabled = true;  // M2: ограничения площади треугольников
    this.areaCompliance = 1e-5;
    this.femEnabled = true;   // M3: коротационный FEM вместо рёбер + площади
    this.young = 1e5;         // модуль Юнга E
    this.nu = 0.3;            // коэффициент Пуассона ν
    this.cutEnabled = true;   // M4: трещины = расщепление вершин (иначе эрозия элементов)
    this.maxSplits = 2;       // максимум расщеплений за кадр (трещина растёт плавно)
    this.splits = 0;          // счётчик расщеплений
    this.substeps = 10;
    this.damping = 0.9995;
    this.tearEnabled = false;
    this.tearStrain = 0.45;   // относительное удлинение, при котором ребро рвётся
    this.bounds = { w: 800, h: 600 };
    this.clear();
  }

  clear() {
    this.n = 0;
    this.x = new Float64Array(0);   // позиции (x,y подряд)
    this.p = new Float64Array(0);   // предыдущие позиции
    this.v = new Float64Array(0);   // скорости
    this.w = new Float64Array(0);   // обратные массы
    this.edges = new Int32Array(0); // пары индексов (a,b)
    this.rest = new Float64Array(0);
    this.alive = new Uint8Array(0);
    this.strain = new Float32Array(0);
    this.tris = new Int32Array(0);  // тройки индексов вершин
    this.triEdges = new Int32Array(0); // тройки индексов рёбер
    this.restArea = new Float64Array(0); // площадь покоя со знаком (знак = ориентация)
    this.dmInv = new Float64Array(0);    // M3: обратная матрица покоя Dm^-1 (4 числа на элемент)
    this.vol = new Float64Array(0);      // M3: площадь покоя |A0| (толщина = 1)
    this.triAlive = new Uint8Array(0);   // M3: жив ли элемент (эрозия при разрыве)
    this.sigma = new Float32Array(0);    // M3: макс. главное напряжение Коши на элементе
    this._f = new Float64Array(8);       // рабочий буфер
    this.cap = 0;                        // M4: ёмкость массивов вершин (n растёт при расщеплении)
    this.sxx = new Float32Array(0);      // M4: тензор напряжения элемента (мировые координаты)
    this.sxy = new Float32Array(0);
    this.syy = new Float32Array(0);
    this.pieceId = new Int32Array(0);    // M4: номер куска для каждого элемента
    this.pieces = 0;                     // M4: число отдельных кусков
    this._acc = new Float64Array(0);
  }

  // Площадь треугольника со знаком (положительная/отрицательная — по ориентации)
  signedArea(t) {
    const x = this.x, a = this.tris[t * 3] * 2, b = this.tris[t * 3 + 1] * 2, c = this.tris[t * 3 + 2] * 2;
    return 0.5 * ((x[b] - x[a]) * (x[c + 1] - x[a + 1]) - (x[c] - x[a]) * (x[b + 1] - x[a + 1]));
  }

  // Сколько треугольников «вывернулось» (знак площади сменился относительно покоя)
  invertedCount() {
    let k = 0;
    for (let t = 0; t < this.restArea.length; t++) {
      if (this.signedArea(t) * this.restArea[t] < 0) k++;
    }
    return k;
  }

  // Прямоугольная плита из nx*ny ячеек, каждая — 2 треугольника (диагонали чередуются)
  buildSlab(nx, ny, spacing, ox, oy, pinLeft) {
    this.clear();
    const cols = nx + 1, rows = ny + 1, n = cols * rows;
    this.n = n;
    const cap = n * 3;  // запас под вершины, которые появятся при расщеплении
    this.cap = cap;
    this.x = new Float64Array(cap * 2); this.p = new Float64Array(cap * 2);
    this.v = new Float64Array(cap * 2); this.w = new Float64Array(cap).fill(1);
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      const k = j * cols + i;
      this.x[k * 2] = this.p[k * 2] = ox + i * spacing;
      this.x[k * 2 + 1] = this.p[k * 2 + 1] = oy + j * spacing;
      if (pinLeft && i === 0) this.w[k] = 0;
    }
    const edgeMap = new Map(), edgeList = [], tris = [], triEdges = [];
    const edgeId = (a, b) => {
      const key = a < b ? a * n + b : b * n + a;
      let id = edgeMap.get(key);
      if (id === undefined) { id = edgeList.length / 2; edgeMap.set(key, id); edgeList.push(a, b); }
      return id;
    };
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const a = j * cols + i, b = a + 1, c = a + cols, d = c + 1;
      const flip = (i + j) % 2 === 0;
      const t = flip ? [[a, b, d], [a, d, c]] : [[a, b, c], [b, d, c]];
      for (const [p0, p1, p2] of t) {
        tris.push(p0, p1, p2);
        triEdges.push(edgeId(p0, p1), edgeId(p1, p2), edgeId(p2, p0));
      }
    }
    this.edges = Int32Array.from(edgeList);
    const m = edgeList.length / 2;
    this.rest = new Float64Array(m);
    for (let e = 0; e < m; e++) {
      const a = this.edges[e * 2], b = this.edges[e * 2 + 1];
      this.rest[e] = Math.hypot(this.x[a * 2] - this.x[b * 2], this.x[a * 2 + 1] - this.x[b * 2 + 1]);
    }
    this.alive = new Uint8Array(m).fill(1);
    this.strain = new Float32Array(m);
    this.tris = Int32Array.from(tris);
    this.triEdges = Int32Array.from(triEdges);
    this.restArea = new Float64Array(tris.length / 3);
    for (let t = 0; t < this.restArea.length; t++) this.restArea[t] = this.signedArea(t);

    // M3: Dm = [X1-X0 | X2-X0], храним Dm^-1
    const T = this.restArea.length;
    this.dmInv = new Float64Array(T * 4);
    this.vol = new Float64Array(T);
    this.triAlive = new Uint8Array(T).fill(1);
    this.sigma = new Float32Array(T);
    for (let t = 0; t < T; t++) {
      const i0 = this.tris[t * 3] * 2, i1 = this.tris[t * 3 + 1] * 2, i2 = this.tris[t * 3 + 2] * 2;
      const a = this.x[i1] - this.x[i0], c = this.x[i1 + 1] - this.x[i0 + 1];
      const b = this.x[i2] - this.x[i0], d = this.x[i2 + 1] - this.x[i0 + 1];
      const det = a * d - b * c;
      this.dmInv.set([d / det, -b / det, -c / det, a / det], t * 4);
      this.vol[t] = Math.abs(det) / 2;
    }
    this.sxx = new Float32Array(T); this.sxy = new Float32Array(T); this.syy = new Float32Array(T);
    this._acc = new Float64Array(cap * 4);
    this.splits = 0;
    this.computePieces();
  }

  // ---------- M3: коротационный FEM (XPBD) ----------
  // F = Ds * Dm^-1;  R — ближайший поворот к F (в 2D замкнутая формула);
  // out = [a, b, c, d, cos, sin, len], где len = |(a+d, c-b)| = tr(R^T F)
  _defRot(t, out) {
    const x = this.x, tr = this.tris, m = this.dmInv;
    const i0 = tr[t * 3] * 2, i1 = tr[t * 3 + 1] * 2, i2 = tr[t * 3 + 2] * 2;
    const dx1 = x[i1] - x[i0], dy1 = x[i1 + 1] - x[i0 + 1];
    const dx2 = x[i2] - x[i0], dy2 = x[i2 + 1] - x[i0 + 1];
    const m00 = m[t * 4], m01 = m[t * 4 + 1], m10 = m[t * 4 + 2], m11 = m[t * 4 + 3];
    const a = dx1 * m00 + dx2 * m10, b = dx1 * m01 + dx2 * m11;
    const c = dy1 * m00 + dy2 * m10, d = dy1 * m01 + dy2 * m11;
    const u1 = a + d, u2 = c - b, len = Math.hypot(u1, u2);
    let cs = 1, sn = 0;
    if (len > 1e-12) { cs = u1 / len; sn = u2 / len; }
    out[0] = a; out[1] = b; out[2] = c; out[3] = d; out[4] = cs; out[5] = sn; out[6] = len;
  }

  // Применить XPBD-коррекцию для ограничения C(F) с градиентом G = dC/dF.
  // Градиенты по частицам: H = G * Dm^-T, ∇x1 = H[:,0], ∇x2 = H[:,1], ∇x0 = -(∇x1+∇x2)
  _apply(t, g00, g01, g10, g11, C, alphaBar) {
    const x = this.x, w = this.w, tr = this.tris, m = this.dmInv;
    const m00 = m[t * 4], m01 = m[t * 4 + 1], m10 = m[t * 4 + 2], m11 = m[t * 4 + 3];
    const h1x = g00 * m00 + g01 * m01, h2x = g00 * m10 + g01 * m11;
    const h1y = g10 * m00 + g11 * m01, h2y = g10 * m10 + g11 * m11;
    const h0x = -(h1x + h2x), h0y = -(h1y + h2y);
    const p0 = tr[t * 3], p1 = tr[t * 3 + 1], p2 = tr[t * 3 + 2];
    const w0 = w[p0], w1 = w[p1], w2 = w[p2];
    const den = w0 * (h0x * h0x + h0y * h0y) + w1 * (h1x * h1x + h1y * h1y) + w2 * (h2x * h2x + h2y * h2y) + alphaBar;
    if (den < 1e-18) return;
    const dl = -C / den;
    x[p0 * 2] += w0 * dl * h0x; x[p0 * 2 + 1] += w0 * dl * h0y;
    x[p1 * 2] += w1 * dl * h1x; x[p1 * 2 + 1] += w1 * dl * h1y;
    x[p2 * 2] += w2 * dl * h2x; x[p2 * 2 + 1] += w2 * dl * h2y;
  }

  _lame() {
    const E = this.young, nu = this.nu;
    return { mu: E / (2 * (1 + nu)), lam: E * nu / (1 - nu * nu) }; // плоское напряжённое состояние
  }

  femSolve(h) {
    const f = this._f, T = this.triAlive.length, vol = this.vol, alive = this.triAlive;
    const { mu, lam } = this._lame();
    const h2 = h * h;
    for (let t = 0; t < T; t++) {
      if (!alive[t]) continue;
      // 1) девиаторная часть: C = ||F - R||_F,  α = 1/(2 μ V0)
      this._defRot(t, f);
      const e00 = f[0] - f[4], e01 = f[1] + f[5], e10 = f[2] - f[5], e11 = f[3] - f[4];
      const C = Math.hypot(e00, e01, e10, e11);
      if (C > 1e-9) this._apply(t, e00 / C, e01 / C, e10 / C, e11 / C, C, 1 / (2 * mu * vol[t]) / h2);
      // 2) объёмная часть: C = tr(R^T F) - 2,  dC/dF = R,  α = 1/(λ V0)
      this._defRot(t, f);
      const C2 = f[6] - 2;
      if (Math.abs(C2) > 1e-12 && lam > 0) this._apply(t, f[4], -f[5], f[5], f[4], C2, 1 / (lam * vol[t]) / h2);
    }
  }

  // Главное напряжение Коши на каждом элементе + эрозия по критерию прочности
  updateStress() {
    const f = this._f, T = this.triAlive.length, sigma = this.sigma, alive = this.triAlive;
    const { mu, lam } = this._lame();
    const crit = this.tearStrain * this.young;
    for (let t = 0; t < T; t++) {
      if (!alive[t]) { sigma[t] = 0; continue; }
      this._defRot(t, f);
      const a = f[0], b = f[1], c = f[2], d = f[3], cs = f[4], sn = f[5];
      // S = R^T F (симметричный фактор растяжения)
      const S00 = cs * a + sn * c, S01 = cs * b + sn * d, S10 = -sn * a + cs * c, S11 = -sn * b + cs * d;
      const s01 = 0.5 * (S01 + S10), mm = 0.5 * (S00 + S11), dd = Math.hypot(0.5 * (S00 - S11), s01);
      const s1 = mm + dd, s2 = mm - dd, trE = s1 + s2 - 2;
      // σ_i = (2μ(s_i-1) + λ tr(S-I)) / s_j   (J = s1 s2)
      const sig1 = (2 * mu * (s1 - 1) + lam * trE) / Math.max(s2, 0.1);
      const sig2 = (2 * mu * (s2 - 1) + lam * trE) / Math.max(s1, 0.1);
      const sm = Math.max(sig1, sig2);
      sigma[t] = sm;
      // тензор напряжения в мировых осях: σ = σ1 w1w1ᵀ + σ2 w2w2ᵀ, w = R·v (v — оси S)
      const phi = 0.5 * Math.atan2(2 * s01, S00 - S11), cp = Math.cos(phi), sp = Math.sin(phi);
      const w1x = cs * cp - sn * sp, w1y = sn * cp + cs * sp;
      const w2x = -(cs * sp + sn * cp), w2y = -sn * sp + cs * cp;
      this.sxx[t] = sig1 * w1x * w1x + sig2 * w2x * w2x;
      this.sxy[t] = sig1 * w1x * w1y + sig2 * w2x * w2y;
      this.syy[t] = sig1 * w1y * w1y + sig2 * w2y * w2y;
      if (this.tearEnabled && !this.cutEnabled && sm > crit) alive[t] = 0; // M3: эрозия
    }
    if (this.tearEnabled && this.cutEnabled) this.cutStep(crit);
  }

  // ---------- M4: разрез сетки ----------
  // Расщепить вершину v плоскостью, перпендикулярной (nx, ny): элементы «впереди» (вдоль n)
  // остаются на v, элементы «позади» получают дубликат вершины. Возвращает индекс новой вершины или -1.
  splitVertex(v, nx, ny) {
    if (this.n >= this.cap || this.w[v] === 0) return -1;
    const tr = this.tris, x = this.x, T = this.triAlive.length;
    const back = [];
    let front = 0;
    for (let t = 0; t < T; t++) {
      if (!this.triAlive[t]) continue;
      const k = tr[t * 3] === v ? 0 : tr[t * 3 + 1] === v ? 1 : tr[t * 3 + 2] === v ? 2 : -1;
      if (k < 0) continue;
      const cx = (x[tr[t * 3] * 2] + x[tr[t * 3 + 1] * 2] + x[tr[t * 3 + 2] * 2]) / 3 - x[v * 2];
      const cy = (x[tr[t * 3] * 2 + 1] + x[tr[t * 3 + 1] * 2 + 1] + x[tr[t * 3 + 2] * 2 + 1]) / 3 - x[v * 2 + 1];
      if (cx * nx + cy * ny < 0) back.push(t * 3 + k); else front++;
    }
    if (!back.length || !front) return -1; // трещине нужны элементы с обеих сторон
    const nv = this.n++;
    this.x[nv * 2] = this.p[nv * 2] = x[v * 2];
    this.x[nv * 2 + 1] = this.p[nv * 2 + 1] = x[v * 2 + 1];
    this.v[nv * 2] = this.v[v * 2]; this.v[nv * 2 + 1] = this.v[v * 2 + 1];
    this.w[nv] = this.w[v];
    for (const slot of back) tr[slot] = nv;
    return nv;
  }

  // Узловое напряжение = среднее по площади тензоров соседних элементов.
  // Вершина с главным напряжением > прочности расщепляется перпендикулярно главному направлению.
  cutStep(crit) {
    const n = this.n, T = this.triAlive.length, tr = this.tris, acc = this._acc;
    acc.fill(0, 0, n * 4);
    for (let t = 0; t < T; t++) {
      if (!this.triAlive[t]) continue;
      const area = Math.abs(this.signedArea(t));
      for (let k = 0; k < 3; k++) {
        const o = tr[t * 3 + k] * 4;
        acc[o] += area * this.sxx[t]; acc[o + 1] += area * this.sxy[t];
        acc[o + 2] += area * this.syy[t]; acc[o + 3] += area;
      }
    }
    const cand = [];
    for (let v = 0; v < n; v++) {
      const A = acc[v * 4 + 3];
      if (A <= 0 || this.w[v] === 0) continue;
      const a = acc[v * 4] / A, b = acc[v * 4 + 1] / A, c = acc[v * 4 + 2] / A;
      const lam = (a + c) / 2 + Math.hypot((a - c) / 2, b);
      if (lam > crit) { const psi = 0.5 * Math.atan2(2 * b, a - c); cand.push([lam, v, Math.cos(psi), Math.sin(psi)]); }
    }
    if (!cand.length) return;
    cand.sort((p, q) => q[0] - p[0]);
    let did = 0;
    for (const [, v, nx, ny] of cand) {
      if (did >= this.maxSplits) break;
      if (this.splitVertex(v, nx, ny) >= 0) did++;
    }
    if (did) { this.splits += did; this.computePieces(); }
  }

  // Связные компоненты: элементы, делящие вершину, принадлежат одному куску (union-find)
  computePieces() {
    const T = this.triAlive.length, tr = this.tris, parent = new Int32Array(this.n);
    for (let i = 0; i < this.n; i++) parent[i] = i;
    const find = a => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    for (let t = 0; t < T; t++) {
      if (!this.triAlive[t]) continue;
      const r0 = find(tr[t * 3]), r1 = find(tr[t * 3 + 1]), r2 = find(tr[t * 3 + 2]);
      parent[r1] = r0; parent[find(r2)] = find(r0);
    }
    const roots = new Set();
    if (this.pieceId.length !== T) this.pieceId = new Int32Array(T);
    for (let t = 0; t < T; t++) {
      if (!this.triAlive[t]) { this.pieceId[t] = -1; continue; }
      const r = find(tr[t * 3]);
      this.pieceId[t] = r; roots.add(r);
    }
    this.pieces = roots.size;
  }

  pick(mx, my, radius) {
    let best = -1, bd = radius * radius;
    for (let i = 0; i < this.n; i++) {
      const dx = this.x[i * 2] - mx, dy = this.x[i * 2 + 1] - my, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  step(dt) {
    const { n, x, p, v, w, edges, rest, alive, strain } = this;
    const h = dt / this.substeps;
    const alpha = this.compliance / (h * h);
    const m = rest.length;
    const maxX = this.bounds.w - 2, maxY = this.bounds.h - 2;

    for (let s = 0; s < this.substeps; s++) {
      // 1. Предсказание позиций
      for (let i = 0; i < n; i++) {
        const ix = i * 2, iy = ix + 1;
        p[ix] = x[ix]; p[iy] = x[iy];
        if (w[i] > 0) {
          v[iy] += this.gravity * h;
          x[ix] += v[ix] * h; x[iy] += v[iy] * h;
        }
      }
      // 2. Ограничения расстояния (одна итерация на подшаг, λ = 0 в начале подшага)
      const fem = this.femEnabled;
      for (let e = 0; e < (fem ? 0 : m); e++) {
        if (!alive[e]) continue;
        const a = edges[e * 2], b = edges[e * 2 + 1];
        const wa = w[a], wb = w[b], ws = wa + wb;
        if (ws === 0) continue;
        let dx = x[a * 2] - x[b * 2], dy = x[a * 2 + 1] - x[b * 2 + 1];
        const len = Math.hypot(dx, dy) || 1e-9;
        const C = len - rest[e];
        const dl = -C / (ws + alpha);       // Δλ в XPBD при λ=0
        dx /= len; dy /= len;
        x[a * 2] += wa * dl * dx; x[a * 2 + 1] += wa * dl * dy;
        x[b * 2] -= wb * dl * dx; x[b * 2 + 1] -= wb * dl * dy;
      }
      // 2б. M2: ограничения площади C = A - A0 (удерживают объём, не дают треугольнику вывернуться)
      if (fem) this.femSolve(h);
      if (!fem && this.areaEnabled) {
        const aAlpha = this.areaCompliance / (h * h);
        const { tris, triEdges, restArea } = this;
        const T = restArea.length;
        for (let t = 0; t < T; t++) {
          if (!alive[triEdges[t * 3]] || !alive[triEdges[t * 3 + 1]] || !alive[triEdges[t * 3 + 2]]) continue;
          const i0 = tris[t * 3], i1 = tris[t * 3 + 1], i2 = tris[t * 3 + 2];
          const x0 = x[i0 * 2], y0 = x[i0 * 2 + 1], x1 = x[i1 * 2], y1 = x[i1 * 2 + 1], x2 = x[i2 * 2], y2 = x[i2 * 2 + 1];
          const C = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) - restArea[t];
          // градиенты площади по вершинам
          const g0x = 0.5 * (y1 - y2), g0y = 0.5 * (x2 - x1);
          const g1x = 0.5 * (y2 - y0), g1y = 0.5 * (x0 - x2);
          const g2x = 0.5 * (y0 - y1), g2y = 0.5 * (x1 - x0);
          const w0 = w[i0], w1 = w[i1], w2 = w[i2];
          const den = w0 * (g0x * g0x + g0y * g0y) + w1 * (g1x * g1x + g1y * g1y) + w2 * (g2x * g2x + g2y * g2y) + aAlpha;
          if (den < 1e-12) continue;
          const dl = -C / den;
          x[i0 * 2] += w0 * dl * g0x; x[i0 * 2 + 1] += w0 * dl * g0y;
          x[i1 * 2] += w1 * dl * g1x; x[i1 * 2 + 1] += w1 * dl * g1y;
          x[i2 * 2] += w2 * dl * g2x; x[i2 * 2 + 1] += w2 * dl * g2y;
        }
      }
      // 3. Столкновения с границами окна
      for (let i = 0; i < n; i++) {
        if (w[i] === 0) continue;
        const ix = i * 2, iy = ix + 1;
        if (x[iy] > maxY) {
          x[iy] = maxY;
          x[ix] = p[ix] + (x[ix] - p[ix]) * 0.7; // трение об пол
        }
        if (x[ix] < 2) x[ix] = 2; else if (x[ix] > maxX) x[ix] = maxX;
      }
      // 4. Скорости из изменения позиций
      for (let i = 0; i < n; i++) {
        if (w[i] === 0) { v[i * 2] = v[i * 2 + 1] = 0; continue; }
        v[i * 2] = (x[i * 2] - p[i * 2]) / h * this.damping;
        v[i * 2 + 1] = (x[i * 2 + 1] - p[i * 2 + 1]) / h * this.damping;
      }
    }

    // Деформация рёбер (для раскраски/разрыва) — раз за кадр
    for (let e = 0; e < m; e++) {
      if (!alive[e]) { strain[e] = 0; continue; }
      const a = edges[e * 2], b = edges[e * 2 + 1];
      const len = Math.hypot(x[a * 2] - x[b * 2], x[a * 2 + 1] - x[b * 2 + 1]);
      const st = Math.abs(len - rest[e]) / rest[e];
      strain[e] = st;
      if (!this.femEnabled && this.tearEnabled && len > rest[e] && st > this.tearStrain) alive[e] = 0;
    }
    if (this.femEnabled) this.updateStress();
  }

  // Кинетическая энергия — для отладки стабильности
  energy() {
    let e = 0;
    for (let i = 0; i < this.n; i++) if (this.w[i] > 0) e += 0.5 / this.w[i] * (this.v[i * 2] ** 2 + this.v[i * 2 + 1] ** 2);
    return e;
  }
}
