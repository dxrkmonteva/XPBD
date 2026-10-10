// Ядро симуляции: 2D XPBD с «малыми шагами» (small steps).
// Позже этот файл переписывается на C++ (тот же интерфейс → WASM).

export class Sim {
  constructor() {
    this.gravity = 900;       // px/s^2
    this.compliance = 1e-5;   // обратная жёсткость рёбер (0 = абсолютно жёсткий)
    this.areaEnabled = true;  // M2: ограничения площади треугольников
    this.areaCompliance = 1e-5;
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
    this.x = new Float64Array(n * 2); this.p = new Float64Array(n * 2);
    this.v = new Float64Array(n * 2); this.w = new Float64Array(n).fill(1);
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
      for (let e = 0; e < m; e++) {
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
      if (this.areaEnabled) {
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
      if (this.tearEnabled && len > rest[e] && st > this.tearStrain) alive[e] = 0;
    }
  }

  // Кинетическая энергия — для отладки стабильности
  energy() {
    let e = 0;
    for (let i = 0; i < this.n; i++) if (this.w[i] > 0) e += 0.5 / this.w[i] * (this.v[i * 2] ** 2 + this.v[i * 2 + 1] ** 2);
    return e;
  }
}
