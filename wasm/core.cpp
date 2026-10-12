// XPBD Fracture Sim — вычислительное ядро на C++ (WebAssembly, без libc и без Emscripten).
//
// Ядро повторяет JS-версию (src/sim.js) операция в операцию: при тех же входных данных скалярная сборка
// даёт те же числа. Топологические операции (разрез, подсчёт кусков) остаются в JS и работают с теми же
// массивами, потому что массивы лежат в линейной памяти WASM, а JS видит их как TypedArray-представления.
//
// Сборка:  wasm/build.sh   (clang --target=wasm32, lld; флаги см. в скрипте)
//   -DUSE_SIMD  — векторные ядра (WASM SIMD, f32x4) + раскраска элементов, см. ниже, M6.2
//
// Связь с JS: JS заполняет два массива параметров (ctx_f64 / ctx_i32): числа и смещения (в байтах) массивов
// в линейной памяти. Индексы ОБЯЗАНЫ совпадать с перечислениями в src/wasm-sim.js.

typedef int i32;
typedef unsigned int u32;
typedef unsigned char u8;
typedef float f32;
typedef double f64;

#define EXPORT(name) __attribute__((export_name(name)))

// ---------- параметры (f64) ----------
enum { P_GRAV, P_COMPL, P_ACOMPL, P_DAMP, P_YOUNG, P_NU, P_TEARSTRAIN, P_BW, P_BH, P_FRIC, P_CELL, P_SMAX, P_VMAX, P_COUNT = 32 }; // P_SMAX — выход: макс. напряжение
// ---------- целые параметры и указатели ----------
enum {
  Q_N, Q_T, Q_M, Q_CAP, Q_SUBSTEPS, Q_FEM, Q_AREA, Q_COLLIDE, Q_TEAR, Q_CUT, Q_PIECES, Q_CONTACTS,
  Q_GRIDCAP, Q_ITEMSCAP, Q_NCOLORS, Q_SIMD,
  Q_X, Q_P, Q_V, Q_W, Q_EDGES, Q_REST, Q_ALIVE, Q_STRAIN, Q_TRIS, Q_TRIEDGES, Q_RESTAREA, Q_DMINV, Q_VOL,
  Q_TRIALIVE, Q_SIGMA, Q_SXX, Q_SXY, Q_SYY, Q_PIECEID, Q_VPIECE, Q_CNT, Q_POS, Q_BB, Q_ITEMS, Q_BND,
  Q_BSTART, Q_BCOUNT, Q_CORDER, Q_CSTART, Q_CMASK, Q_ORD, Q_TP, Q_DIRTY, Q_COUNT = 64
};

static f64 P[P_COUNT];
static i32 Q[Q_COUNT];

extern "C" {
EXPORT("ctx_f64") f64* ctx_f64() { return P; }
EXPORT("ctx_i32") i32* ctx_i32() { return Q; }
}

// Локальная копия указателей на время вызова step (чтобы компилятор не перечитывал Q[] в циклах)
struct Ctx {
  i32 n, T, m, cap;
  f64 *x, *p, *v, *w, *rest, *restArea, *dmInv, *vol;
  i32 *edges, *tris, *triEdges, *pieceId, *vPiece, *cnt, *pos, *bb, *items, *bnd, *bStart, *bCount;
  u8 *alive, *triAlive;
  f32 *strain, *sigma, *sxx, *sxy, *syy;
};
static Ctx C;

template <class T> static inline T* ptr(i32 idx) { return (T*)(unsigned long)(u32)Q[idx]; }

static void load_ctx() {
  C.n = Q[Q_N]; C.T = Q[Q_T]; C.m = Q[Q_M]; C.cap = Q[Q_CAP];
  C.x = ptr<f64>(Q_X); C.p = ptr<f64>(Q_P); C.v = ptr<f64>(Q_V); C.w = ptr<f64>(Q_W);
  C.rest = ptr<f64>(Q_REST); C.restArea = ptr<f64>(Q_RESTAREA); C.dmInv = ptr<f64>(Q_DMINV); C.vol = ptr<f64>(Q_VOL);
  C.edges = ptr<i32>(Q_EDGES); C.tris = ptr<i32>(Q_TRIS); C.triEdges = ptr<i32>(Q_TRIEDGES);
  C.pieceId = ptr<i32>(Q_PIECEID); C.vPiece = ptr<i32>(Q_VPIECE);
  C.cnt = ptr<i32>(Q_CNT); C.pos = ptr<i32>(Q_POS); C.bb = ptr<i32>(Q_BB); C.items = ptr<i32>(Q_ITEMS);
  C.bnd = ptr<i32>(Q_BND); C.bStart = ptr<i32>(Q_BSTART); C.bCount = ptr<i32>(Q_BCOUNT);
  C.alive = ptr<u8>(Q_ALIVE); C.triAlive = ptr<u8>(Q_TRIALIVE);
  C.strain = ptr<f32>(Q_STRAIN); C.sigma = ptr<f32>(Q_SIGMA);
  C.sxx = ptr<f32>(Q_SXX); C.sxy = ptr<f32>(Q_SXY); C.syy = ptr<f32>(Q_SYY);
}

static inline f64 hyp(f64 a, f64 b) { return __builtin_sqrt(a * a + b * b); }

// ================= M3: коротационный FEM (XPBD) =================
// F = Ds * Dm^-1; R — ближайший поворот (2D, замкнутая формула). out = [a, b, c, d, cos, sin, tr(R^T F)]
static inline void defRot(i32 t, f64* out) {
  const f64* x = C.x; const i32* tr = C.tris; const f64* m = C.dmInv;
  const i32 i0 = tr[t * 3] * 2, i1 = tr[t * 3 + 1] * 2, i2 = tr[t * 3 + 2] * 2;
  const f64 dx1 = x[i1] - x[i0], dy1 = x[i1 + 1] - x[i0 + 1];
  const f64 dx2 = x[i2] - x[i0], dy2 = x[i2 + 1] - x[i0 + 1];
  const f64 m00 = m[t * 4], m01 = m[t * 4 + 1], m10 = m[t * 4 + 2], m11 = m[t * 4 + 3];
  const f64 a = dx1 * m00 + dx2 * m10, b = dx1 * m01 + dx2 * m11;
  const f64 c = dy1 * m00 + dy2 * m10, d = dy1 * m01 + dy2 * m11;
  const f64 u1 = a + d, u2 = c - b, len = hyp(u1, u2);
  f64 cs = 1, sn = 0;
  if (len > 1e-12) { cs = u1 / len; sn = u2 / len; }
  out[0] = a; out[1] = b; out[2] = c; out[3] = d; out[4] = cs; out[5] = sn; out[6] = len;
}

// XPBD-коррекция для ограничения C(F) с градиентом G = dC/dF; ∇x = G * Dm^-T
static inline void applyC(i32 t, f64 g00, f64 g01, f64 g10, f64 g11, f64 Cv, f64 alphaBar) {
  f64* x = C.x; const f64* w = C.w; const i32* tr = C.tris; const f64* m = C.dmInv;
  const f64 m00 = m[t * 4], m01 = m[t * 4 + 1], m10 = m[t * 4 + 2], m11 = m[t * 4 + 3];
  const f64 h1x = g00 * m00 + g01 * m01, h2x = g00 * m10 + g01 * m11;
  const f64 h1y = g10 * m00 + g11 * m01, h2y = g10 * m10 + g11 * m11;
  const f64 h0x = -(h1x + h2x), h0y = -(h1y + h2y);
  const i32 p0 = tr[t * 3], p1 = tr[t * 3 + 1], p2 = tr[t * 3 + 2];
  const f64 w0 = w[p0], w1 = w[p1], w2 = w[p2];
  const f64 den = w0 * (h0x * h0x + h0y * h0y) + w1 * (h1x * h1x + h1y * h1y) + w2 * (h2x * h2x + h2y * h2y) + alphaBar;
  if (den < 1e-18) return;
  const f64 dl = -Cv / den;
  x[p0 * 2] += w0 * dl * h0x; x[p0 * 2 + 1] += w0 * dl * h0y;
  x[p1 * 2] += w1 * dl * h1x; x[p1 * 2 + 1] += w1 * dl * h1y;
  x[p2 * 2] += w2 * dl * h2x; x[p2 * 2 + 1] += w2 * dl * h2y;
}

static inline f64 muOf() { return P[P_YOUNG] / (2 * (1 + P[P_NU])); }
static inline f64 lamOf() { return P[P_YOUNG] * P[P_NU] / (1 - P[P_NU] * P[P_NU]); } // плоское напряжённое состояние

static void femSolveScalar(f64 h) {
  f64 f[8];
  const f64 mu = muOf(), lam = lamOf(), h2 = h * h;
  for (i32 t = 0; t < C.T; t++) {
    if (!C.triAlive[t]) continue;
    defRot(t, f);
    const f64 e00 = f[0] - f[4], e01 = f[1] + f[5], e10 = f[2] - f[5], e11 = f[3] - f[4];
    const f64 Cd = __builtin_sqrt(e00 * e00 + e01 * e01 + e10 * e10 + e11 * e11);
    if (Cd > 1e-9) applyC(t, e00 / Cd, e01 / Cd, e10 / Cd, e11 / Cd, Cd, 1 / (2 * mu * C.vol[t]) / h2);
    defRot(t, f);
    const f64 C2 = f[6] - 2;
    if (__builtin_fabs(C2) > 1e-12 && lam > 0) applyC(t, f[4], -f[5], f[5], f[4], C2, 1 / (lam * C.vol[t]) / h2);
  }
}


#ifdef USE_SIMD
#include <wasm_simd128.h>
#endif

// ================= M7: потоки (WebAssembly threads) =================
// Рабочие потоки (Web Workers) делят с главным ту же линейную память и крутятся в worker_loop. Главный поток в step()
// раздаёт «параллельные задачи» (parallel-for): записывает параметры, увеличивает счётчик поколений G_gen, и все потоки
// обрабатывают свои доли; главный поток тоже работает (доля 0) и ждёт счётчик G_done. Элементы одного цвета не делят
// вершин, поэтому записи потоков не пересекаются. Главный поток в браузере не может блокироваться — он только
// крутится в цикле ожидания; рабочие после паузы засыпают (atomic.wait), чтобы не греть процессор между кадрами.
enum { JOB_FEM = 1, JOB_STRESS = 2 };
__attribute__((unused)) static void femColorRange(i32 col, i32 g0, i32 g1);
static f64 stressRange(i32 t0, i32 t1);
__attribute__((unused)) static i32 G_gen, G_done, G_quit, G_sleepers, G_kind, G_col, G_ng;
static i32 G_nth = 1;
static f64 G_smax[16];

#ifdef USE_THREADS
static const i32 PAR_MIN_GROUPS = 48, PAR_MIN_TRIS = 4096, SPIN_LIMIT = 20000;
static inline bool parOn() { return G_nth > 1; }

// Динамическое распределение: потоки берут куски по очереди (атомарный счётчик G_next), поэтому медленные ядра
// (big.LITTLE на телефонах) не тормозят остальные — просто делают меньше кусков.
static i32 G_next;
static void runSlice(i32 kind, i32 tid, i32 /*nth*/) {
  if (kind == JOB_FEM) {
    const i32 CH = 16;
    for (;;) {
      const i32 lo = __atomic_fetch_add(&G_next, CH, __ATOMIC_SEQ_CST);
      if (lo >= G_ng) break;
      femColorRange(G_col, lo, lo + CH < G_ng ? lo + CH : G_ng);
    }
  } else if (kind == JOB_STRESS) {
    const i32 CH = 512; f64 sm = 0;
    for (;;) {
      const i32 lo = __atomic_fetch_add(&G_next, CH, __ATOMIC_SEQ_CST);
      if (lo >= C.T) break;
      const f64 v = stressRange(lo, lo + CH < C.T ? lo + CH : C.T);
      if (v > sm) sm = v;
    }
    G_smax[tid] = sm;
  }
}

static void par(i32 kind) {
  const i32 nth = G_nth;
  __atomic_store_n(&G_done, 0, __ATOMIC_SEQ_CST);
  G_kind = kind;
  __atomic_store_n(&G_next, 0, __ATOMIC_SEQ_CST);
  __atomic_add_fetch(&G_gen, 1, __ATOMIC_SEQ_CST);
  if (__atomic_load_n(&G_sleepers, __ATOMIC_SEQ_CST) > 0) __builtin_wasm_memory_atomic_notify(&G_gen, 1000);
  runSlice(kind, 0, nth);
  while (__atomic_load_n(&G_done, __ATOMIC_SEQ_CST) < nth - 1) { }
}

extern "C" {
EXPORT("set_threads") void set_threads(i32 n) { G_nth = n < 1 ? 1 : (n > 16 ? 16 : n); G_quit = 0; }
EXPORT("stop_workers") void stop_workers() { __atomic_store_n(&G_quit, 1, __ATOMIC_SEQ_CST); __builtin_wasm_memory_atomic_notify(&G_gen, 1000); }
EXPORT("worker_loop") void worker_loop(i32 tid) {
  i32 seen = __atomic_load_n(&G_gen, __ATOMIC_SEQ_CST);
  for (;;) {
    i32 g, spins = 0;
    while ((g = __atomic_load_n(&G_gen, __ATOMIC_SEQ_CST)) == seen) {
      if (__atomic_load_n(&G_quit, __ATOMIC_SEQ_CST)) return;
      // лишние потоки (tid >= G_nth) и потоки, которым долго не дают работы, спят до уведомления:
      // иначе они крутятся в цикле, отнимают ядра у главного потока и греют телефон
      if (tid >= G_nth || ++spins > SPIN_LIMIT) {
        __atomic_add_fetch(&G_sleepers, 1, __ATOMIC_SEQ_CST);
        __builtin_wasm_memory_atomic_wait32(&G_gen, seen, -1);
        __atomic_sub_fetch(&G_sleepers, 1, __ATOMIC_SEQ_CST);
        spins = SPIN_LIMIT; // после пробуждения без новой работы сразу снова спим
      }
    }
    seen = g;
    const i32 nth = G_nth;
    if (tid < nth) { runSlice(G_kind, tid, nth); __atomic_add_fetch(&G_done, 1, __ATOMIC_SEQ_CST); }
  }
}
}
#else
static inline bool parOn() { return false; }
__attribute__((unused)) static const i32 PAR_MIN_GROUPS = 1 << 30; static const i32 PAR_MIN_TRIS = 1 << 30;
static void par(i32) {}
#endif

#ifdef USE_SIMD
// ---------- M6.2: векторное FEM-ядро (WASM SIMD, f32x4) + раскраска элементов ----------
// Элементы одного цвета не делят вершин, поэтому 4 элемента можно обрабатывать одновременно (Гаусс–Зейдель по цветам).
// Позиции и разности рёбер считаются в f64 (точность), тяжёлая математика (sqrt, деление, градиенты) — в f32x4,
// поправки применяются к позициям в f64. Gather/scatter скалярные: в WASM SIMD нет векторных gather/scatter.
// f4 — настоящий векторный тип float x4 (у v128_t операторы работают как у int64x2, поэтому приводим типы явно)
typedef float f4 __attribute__((vector_size(16)));
typedef int m4 __attribute__((vector_size(16)));
#define F4(x) ((f4){(x), (x), (x), (x)})
#define LD(p) ((f4)wasm_v128_load(p))
#define ST(p, v) wasm_v128_store((p), (v128_t)(v))
#define SQRT4(v) ((f4)wasm_f32x4_sqrt((v128_t)(v)))
#define ABS4(v) ((f4)wasm_f32x4_abs((v128_t)(v)))
struct Def4 { f4 a, b, c, d, cs, sn, len; };
struct Del4 { f4 x0, y0, x1, y1, x2, y2; }; // поправки к трём вершинам

static inline f4 sel(m4 m, f4 a, f4 b) { return (f4)wasm_v128_bitselect((v128_t)a, (v128_t)b, (v128_t)m); }

// Деформация 4 элементов по векторам рёбер (dx1,dy1) = x1-x0 и (dx2,dy2) = x2-x0
static inline void defRot4(f4 X1, f4 Y1, f4 X2, f4 Y2, f4 m00, f4 m01, f4 m10, f4 m11, Def4& o) {
  o.a = X1 * m00 + X2 * m10; o.b = X1 * m01 + X2 * m11;
  o.c = Y1 * m00 + Y2 * m10; o.d = Y1 * m01 + Y2 * m11;
  const f4 u1 = o.a + o.d, u2 = o.c - o.b;
  o.len = SQRT4(u1 * u1 + u2 * u2);
  const m4 ok = o.len > F4(1e-12f);
  const f4 inv = F4(1.0f) / sel(ok, o.len, F4(1.0f));
  o.cs = sel(ok, u1 * inv, F4(1.0f)); o.sn = sel(ok, u2 * inv, F4(0.0f));
}

// Поправка XPBD для ограничения с градиентом G (как applyC), без записи в память
static inline void deltaC4(f4 m00, f4 m01, f4 m10, f4 m11, f4 w0, f4 w1, f4 w2,
                           f4 g00, f4 g01, f4 g10, f4 g11, f4 Cv, f4 alphaBar, m4 active, Del4& d) {
  const f4 h1x = g00 * m00 + g01 * m01, h2x = g00 * m10 + g01 * m11;
  const f4 h1y = g10 * m00 + g11 * m01, h2y = g10 * m10 + g11 * m11;
  const f4 h0x = -(h1x + h2x), h0y = -(h1y + h2y);
  const f4 den = w0 * (h0x * h0x + h0y * h0y) + w1 * (h1x * h1x + h1y * h1y) + w2 * (h2x * h2x + h2y * h2y) + alphaBar;
  const m4 ok = active & (den > F4(1e-18f));
  const f4 dl = sel(ok, -Cv / sel(ok, den, F4(1.0f)), F4(0.0f));
  d.x0 = w0 * dl * h0x; d.y0 = w0 * dl * h0y;
  d.x1 = w1 * dl * h1x; d.y1 = w1 * dl * h1y;
  d.x2 = w2 * dl * h2x; d.y2 = w2 * dl * h2y;
}

// Данные элементов в порядке раскраски (SoA, группами по 4, хвост группы дополнен пустыми элементами):
// v0,v1,v2 (смещения вершин в f64-массиве x), m00..m11 (Dm^-1, f32), 1/V, w0,w1,w2 (обратные массы вершин), act.
// Статическая часть (индексы, Dm^-1, 1/V) пересчитывается при смене топологии (Q_DIRTY), динамическая (w, act) —
// раз за кадр: так горячий цикл читает всё подряд, а в памяти «случайно» остаются только позиции вершин.
enum { O_V0, O_V1, O_V2, O_M00, O_M01, O_M10, O_M11, O_IV, O_W0, O_W1, O_W2, O_ACT, O_N };

static void prepareColored() {
  const i32 Tp = Q[Q_TP]; char* base = ptr<char>(Q_ORD);
  const i32* order = ptr<i32>(Q_CORDER); const i32* tr = C.tris; const f64* w = C.w;
  i32* ov[3] = {(i32*)base, (i32*)base + Tp, (i32*)base + 2 * Tp};
  f32* of = (f32*)base;
  if (Q[Q_DIRTY]) {
    for (i32 i = 0; i < Tp; i++) {
      const i32 t = order[i];
      if (t < 0) {
        ov[0][i] = ov[1][i] = ov[2][i] = 0;
        for (i32 q = 0; q < 4; q++) of[(O_M00 + q) * Tp + i] = 0;
        of[O_IV * Tp + i] = 0;
        continue;
      }
      ov[0][i] = tr[t * 3] * 2; ov[1][i] = tr[t * 3 + 1] * 2; ov[2][i] = tr[t * 3 + 2] * 2;
      for (i32 q = 0; q < 4; q++) of[(O_M00 + q) * Tp + i] = (f32)C.dmInv[t * 4 + q];
      of[O_IV * Tp + i] = (f32)(1.0 / C.vol[t]);
    }
    Q[Q_DIRTY] = 0;
  }
  for (i32 i = 0; i < Tp; i++) {
    const i32 t = order[i];
    if (t < 0) { of[O_ACT * Tp + i] = 0; of[O_W0 * Tp + i] = of[O_W1 * Tp + i] = of[O_W2 * Tp + i] = 0; continue; }
    of[O_ACT * Tp + i] = C.triAlive[t] ? 1.0f : 0.0f;
    of[O_W0 * Tp + i] = (f32)w[tr[t * 3]]; of[O_W1 * Tp + i] = (f32)w[tr[t * 3 + 1]]; of[O_W2 * Tp + i] = (f32)w[tr[t * 3 + 2]];
  }
}

// Раскрашенный Гаусс–Зейдель. Для группы из 4 элементов позиции читаются ОДИН раз; обе поправки (девиаторная и
// объёмная) считаются по векторам рёбер в регистрах (вторая — с учётом первой), в память пишется суммарная поправка.
static f64 G_h;
static void femColorRange(i32 col, i32 g0, i32 g1) {
  const f64 mu = muOf(), lam = lamOf(), h2 = G_h * G_h;
  const i32* cstart = ptr<i32>(Q_CSTART);
  const i32 Tp = Q[Q_TP]; char* base = ptr<char>(Q_ORD);
  const i32* iv0 = (const i32*)base; const i32* iv1 = iv0 + Tp; const i32* iv2 = iv1 + Tp;
  const f32* of = (const f32*)base;
  const f4 alphaD0 = F4((f32)(1 / (2 * mu) / h2)), alphaV0 = F4((f32)(lam > 0 ? 1 / lam / h2 : 0));
  const m4 lamOn = F4((f32)lam) > F4(0.0f);
  f64* xm = C.x;
  {
    for (i32 k = cstart[col] + 4 * g0; k < cstart[col] + 4 * g1; k += 4) {
      const f4 act4 = LD(of + O_ACT * Tp + k);
      if (!wasm_v128_any_true((v128_t)(act4 > F4(0.5f)))) continue;
      // gather: позиции вершин (f64) → векторы рёбер (f32); сборка через make/replace_lane — без обращения к памяти
      const i32 *I0 = iv0 + k, *I1 = iv1 + k, *I2 = iv2 + k;
#define EDGE(l, X, Y) ((f32)(xm[X[l]] - xm[I0[l]])), ((f32)(xm[X[l] + 1] - xm[I0[l] + 1]))
      f4 X1 = (f4)wasm_f32x4_make((f32)(xm[I1[0]] - xm[I0[0]]), (f32)(xm[I1[1]] - xm[I0[1]]), (f32)(xm[I1[2]] - xm[I0[2]]), (f32)(xm[I1[3]] - xm[I0[3]]));
      f4 Y1 = (f4)wasm_f32x4_make((f32)(xm[I1[0] + 1] - xm[I0[0] + 1]), (f32)(xm[I1[1] + 1] - xm[I0[1] + 1]), (f32)(xm[I1[2] + 1] - xm[I0[2] + 1]), (f32)(xm[I1[3] + 1] - xm[I0[3] + 1]));
      f4 X2 = (f4)wasm_f32x4_make((f32)(xm[I2[0]] - xm[I0[0]]), (f32)(xm[I2[1]] - xm[I0[1]]), (f32)(xm[I2[2]] - xm[I0[2]]), (f32)(xm[I2[3]] - xm[I0[3]]));
      f4 Y2 = (f4)wasm_f32x4_make((f32)(xm[I2[0] + 1] - xm[I0[0] + 1]), (f32)(xm[I2[1] + 1] - xm[I0[1] + 1]), (f32)(xm[I2[2] + 1] - xm[I0[2] + 1]), (f32)(xm[I2[3] + 1] - xm[I0[3] + 1]));
#undef EDGE
      const m4 act = act4 > F4(0.5f);
      const f4 m00 = LD(of + O_M00 * Tp + k), m01 = LD(of + O_M01 * Tp + k), m10 = LD(of + O_M10 * Tp + k), m11 = LD(of + O_M11 * Tp + k);
      const f4 iV = LD(of + O_IV * Tp + k), w0 = LD(of + O_W0 * Tp + k), w1 = LD(of + O_W1 * Tp + k), w2 = LD(of + O_W2 * Tp + k);
      Def4 d; Del4 q1, q2;
      defRot4(X1, Y1, X2, Y2, m00, m01, m10, m11, d);
      const f4 e00 = d.a - d.cs, e01 = d.b + d.sn, e10 = d.c - d.sn, e11 = d.d - d.cs;
      const f4 Cd = SQRT4(e00 * e00 + e01 * e01 + e10 * e10 + e11 * e11);
      const m4 okd = act & (Cd > F4(1e-9f));
      const f4 invC = F4(1.0f) / sel(okd, Cd, F4(1.0f));
      deltaC4(m00, m01, m10, m11, w0, w1, w2, e00 * invC, e01 * invC, e10 * invC, e11 * invC, Cd, alphaD0 * iV, okd, q1);
      X1 += q1.x1 - q1.x0; Y1 += q1.y1 - q1.y0; X2 += q1.x2 - q1.x0; Y2 += q1.y2 - q1.y0;
      defRot4(X1, Y1, X2, Y2, m00, m01, m10, m11, d);
      const f4 C2 = d.len - F4(2.0f);
      const m4 okv = act & (ABS4(C2) > F4(1e-12f)) & lamOn;
      deltaC4(m00, m01, m10, m11, w0, w1, w2, d.cs, -d.sn, d.sn, d.cs, C2, alphaV0 * iV, okv, q2);
      const f4 Ax = q1.x0 + q2.x0, Ay = q1.y0 + q2.y0, Bx = q1.x1 + q2.x1, By = q1.y1 + q2.y1, Cx = q1.x2 + q2.x2, Cy = q1.y2 + q2.y2;
#define SCAT(l) { const i32 a0 = I0[l], a1 = I1[l], a2 = I2[l]; \
        xm[a0] += wasm_f32x4_extract_lane((v128_t)Ax, l); xm[a0 + 1] += wasm_f32x4_extract_lane((v128_t)Ay, l); \
        xm[a1] += wasm_f32x4_extract_lane((v128_t)Bx, l); xm[a1 + 1] += wasm_f32x4_extract_lane((v128_t)By, l); \
        xm[a2] += wasm_f32x4_extract_lane((v128_t)Cx, l); xm[a2 + 1] += wasm_f32x4_extract_lane((v128_t)Cy, l); }
      SCAT(0) SCAT(1) SCAT(2) SCAT(3)
#undef SCAT
    }
  }
}

static void femSolveColored(f64 h) {
  G_h = h;
  const i32* cstart = ptr<i32>(Q_CSTART);
  for (i32 col = 0; col < Q[Q_NCOLORS]; col++) {
    const i32 ng = (cstart[col + 1] - cstart[col]) / 4;
    if (parOn() && ng >= PAR_MIN_GROUPS) { G_col = col; G_ng = ng; par(JOB_FEM); }
    else femColorRange(col, 0, ng);
  }
}
#endif

// Главное напряжение Коши на элементе + тензор в мировых осях; эрозия (если трещины выключены)
static f64 stressRange(i32 t0, i32 t1) {
  f64 f[8];
  const f64 mu = muOf(), lam = lamOf();
  const f64 crit = P[P_TEARSTRAIN] * P[P_YOUNG];
  const bool erode = Q[Q_TEAR] && !Q[Q_CUT];
  f64 smax = 0;
  for (i32 t = t0; t < t1; t++) {
    if (!C.triAlive[t]) { C.sigma[t] = 0; continue; }
    defRot(t, f);
    const f64 a = f[0], b = f[1], c = f[2], d = f[3], cs = f[4], sn = f[5];
    const f64 S00 = cs * a + sn * c, S01 = cs * b + sn * d, S10 = -sn * a + cs * c, S11 = -sn * b + cs * d;
    const f64 s01 = 0.5 * (S01 + S10), mm = 0.5 * (S00 + S11), dd = hyp(0.5 * (S00 - S11), s01);
    const f64 s1 = mm + dd, s2 = mm - dd, trE = s1 + s2 - 2;
    const f64 sig1 = (2 * mu * (s1 - 1) + lam * trE) / (s2 > 0.1 ? s2 : 0.1);
    const f64 sig2 = (2 * mu * (s2 - 1) + lam * trE) / (s1 > 0.1 ? s1 : 0.1);
    const f64 sm = sig1 > sig2 ? sig1 : sig2;
    C.sigma[t] = (f32)sm;
    if (sm > smax) smax = sm;
    const f64 hu = 0.5 * (S00 - S11);
    f64 cp = 1, sp = 0;
    if (dd > 0) { cp = __builtin_sqrt(0.5 * (1 + hu / dd)); sp = __builtin_sqrt(0.5 * (1 - hu / dd)); if (s01 < 0) sp = -sp; }
    const f64 w1x = cs * cp - sn * sp, w1y = sn * cp + cs * sp;
    const f64 w2x = -(cs * sp + sn * cp), w2y = -sn * sp + cs * cp;
    C.sxx[t] = (f32)(sig1 * w1x * w1x + sig2 * w2x * w2x);
    C.sxy[t] = (f32)(sig1 * w1x * w1y + sig2 * w2x * w2y);
    C.syy[t] = (f32)(sig1 * w1y * w1y + sig2 * w2y * w2y);
    if (erode && sm > crit) C.triAlive[t] = 0;
  }
  return smax;
}

static void updateStress() {
  f64 smax = 0;
  if (parOn() && C.T >= PAR_MIN_TRIS) {
    par(JOB_STRESS);
    for (i32 i = 0; i < G_nth; i++) if (G_smax[i] > smax) smax = G_smax[i];
  } else smax = stressRange(0, C.T);
  P[P_SMAX] = smax;
}

// ================= M5: столкновения между кусками =================
static inline i32 clampCell(f64 v, i32 hi) {
  if (!(v >= 0)) return 0;
  if (v > hi) return hi;
  return (i32)v;
}

// Вершина (px,py) строго внутри элемента t (с допуском 1e-4)?
static inline bool insideTri(f64 px, f64 py, i32 t) {
  const f64* x = C.x; const i32* tr = C.tris;
  const f64 ra = C.restArea[t];
  const f64 s = ra > 0 ? 1 : -1;
  const i32 a0 = tr[t * 3] * 2, b0 = tr[t * 3 + 1] * 2, c0 = tr[t * 3 + 2] * 2;
  const f64 sa = 0.5 * ((x[b0] - x[a0]) * (x[c0 + 1] - x[a0 + 1]) - (x[c0] - x[a0]) * (x[b0 + 1] - x[a0 + 1]));
  if (sa * ra <= 0) return false; // вывернутый элемент не участвует
  for (i32 e = 0; e < 3; e++) {
    const i32 a = tr[t * 3 + e] * 2, b = tr[t * 3 + (e + 1) % 3] * 2;
    const f64 ex = x[b] - x[a], ey = x[b + 1] - x[a + 1];
    f64 len = hyp(ex, ey);
    if (len == 0) len = 1e-9;
    const f64 d = s * (ex * (py - x[a + 1]) - ey * (px - x[a])) / len;
    if (d <= 1e-4) return false;
  }
  return true;
}

static void collide() {
  const i32 T = C.T, n = C.n;
  const f64 cs = P[P_CELL];
  const i32 gw = (i32)__builtin_ceil(P[P_BW] / cs) + 2, gh = (i32)__builtin_ceil(P[P_BH] / cs) + 2, G = gw * gh;
  Q[Q_CONTACTS] = 0;
  if (G > Q[Q_GRIDCAP]) return;
  i32 *cnt = C.cnt, *pos = C.pos, *bb = C.bb, *items = C.items;
  const f64* x = C.x; const i32* tr = C.tris;
  for (i32 c = 0; c <= G; c++) cnt[c] = 0;
  for (i32 t = 0; t < T; t++) {
    if (!C.triAlive[t]) { bb[t * 4] = 1; bb[t * 4 + 2] = 0; continue; } // пустой диапазон
    const i32 a = tr[t * 3] * 2, b = tr[t * 3 + 1] * 2, c = tr[t * 3 + 2] * 2;
    f64 x0 = x[a], x1 = x[a], y0 = x[a + 1], y1 = x[a + 1];
    if (x[b] < x0) x0 = x[b]; if (x[c] < x0) x0 = x[c];
    if (x[b] > x1) x1 = x[b]; if (x[c] > x1) x1 = x[c];
    if (x[b + 1] < y0) y0 = x[b + 1]; if (x[c + 1] < y0) y0 = x[c + 1];
    if (x[b + 1] > y1) y1 = x[b + 1]; if (x[c + 1] > y1) y1 = x[c + 1];
    x0 -= 1; x1 += 1; y0 -= 1; y1 += 1;
    const i32 ix0 = clampCell(__builtin_floor(x0 / cs), gw - 1), ix1 = clampCell(__builtin_floor(x1 / cs), gw - 1);
    const i32 iy0 = clampCell(__builtin_floor(y0 / cs), gh - 1), iy1 = clampCell(__builtin_floor(y1 / cs), gh - 1);
    bb[t * 4] = ix0; bb[t * 4 + 1] = iy0; bb[t * 4 + 2] = ix1; bb[t * 4 + 3] = iy1;
    for (i32 iy = iy0; iy <= iy1; iy++) for (i32 ix = ix0; ix <= ix1; ix++) cnt[iy * gw + ix + 1]++;
  }
  for (i32 c = 1; c <= G; c++) cnt[c] += cnt[c - 1];
  if (cnt[G] > Q[Q_ITEMSCAP]) return; // слишком много пар (куски «взорвались») — пропускаем контакты в этом подшаге
  for (i32 c = 0; c < G; c++) pos[c] = cnt[c];
  for (i32 t = 0; t < T; t++) {
    if (!C.triAlive[t]) continue;
    for (i32 iy = bb[t * 4 + 1]; iy <= bb[t * 4 + 3]; iy++)
      for (i32 ix = bb[t * 4]; ix <= bb[t * 4 + 2]; ix++) items[pos[iy * gw + ix]++] = t;
  }

  i32 contacts = 0;
  const f64 skin = 0.02, mu = P[P_FRIC];
  f64* xm = C.x; const f64* p = C.p; const f64* w = C.w;
  const i32 *pid = C.pieceId, *vp = C.vPiece;
  for (i32 v = 0; v < n; v++) {
    if (vp[v] < 0 || w[v] == 0) continue;
    const f64 px = xm[v * 2], py = xm[v * 2 + 1];
    const i32 cell = clampCell(__builtin_floor(py / cs), gh - 1) * gw + clampCell(__builtin_floor(px / cs), gw - 1);
    for (i32 k = cnt[cell]; k < cnt[cell + 1]; k++) {
      const i32 t = items[k];
      if (pid[t] == vp[v]) continue;
      if (!insideTri(px, py, t)) continue;
      // ближайшая точка на ГРАНИЦЕ чужого куска
      const i32 pc = pid[t], bc = C.bCount[pc];
      if (!bc) continue;
      const i32* barr = C.bnd + C.bStart[pc] * 3;
      i32 ia = -1, ib = -1, bq = 0;
      f64 s = 0, bd = 1.0 / 0.0, cx = 0, cy = 0;
      for (i32 q = 0; q < bc; q++) {
        const i32 a0 = barr[q * 3] * 2, b0 = barr[q * 3 + 1] * 2;
        const f64 ex = xm[b0] - xm[a0], ey = xm[b0 + 1] - xm[a0 + 1];
        f64 l2 = ex * ex + ey * ey;
        if (l2 == 0) l2 = 1e-12;
        f64 ss = ((px - xm[a0]) * ex + (py - xm[a0 + 1]) * ey) / l2;
        ss = ss > 1 ? 1 : ss;
        ss = ss < 0 ? 0 : ss;
        const f64 qx = xm[a0] + ss * ex, qy = xm[a0 + 1] + ss * ey;
        const f64 dd = (px - qx) * (px - qx) + (py - qy) * (py - qy);
        if (dd < bd) { bd = dd; ia = barr[q * 3]; ib = barr[q * 3 + 1]; s = ss; cx = qx; cy = qy; bq = q; }
      }
      if (ia < 0) continue;
      const f64 dist = __builtin_sqrt(bd);
      f64 nx, ny;
      if (dist > 1e-6) { nx = (cx - px) / dist; ny = (cy - py) / dist; }
      else {
        const i32 o = barr[bq * 3 + 2], a1 = ia * 2;
        const f64 ex = xm[ib * 2] - xm[a1], ey = xm[ib * 2 + 1] - xm[a1 + 1];
        f64 l = hyp(ex, ey);
        if (l == 0) l = 1e-9;
        i32 c3 = tr[o * 3]; if (c3 == ia || c3 == ib) c3 = tr[o * 3 + 1]; if (c3 == ia || c3 == ib) c3 = tr[o * 3 + 2];
        nx = ey / l; ny = -ex / l;
        if (nx * (xm[c3 * 2] - xm[a1]) + ny * (xm[c3 * 2 + 1] - xm[a1 + 1]) > 0) { nx = -nx; ny = -ny; }
      }
      const f64 wa = w[ia], wb = w[ib];
      const f64 den = w[v] + (1 - s) * (1 - s) * wa + s * s * wb;
      if (den < 1e-12) continue;
      const f64 pen = dist + skin, lam = pen / den;
      xm[v * 2] += w[v] * lam * nx; xm[v * 2 + 1] += w[v] * lam * ny;
      xm[ia * 2] -= wa * (1 - s) * lam * nx; xm[ia * 2 + 1] -= wa * (1 - s) * lam * ny;
      xm[ib * 2] -= wb * s * lam * nx; xm[ib * 2 + 1] -= wb * s * lam * ny;
      // трение Кулона: гасим относительное касательное смещение (в пределах μ·pen)
      const f64 rx = (xm[v * 2] - p[v * 2]) - ((1 - s) * (xm[ia * 2] - p[ia * 2]) + s * (xm[ib * 2] - p[ib * 2]));
      const f64 ry = (xm[v * 2 + 1] - p[v * 2 + 1]) - ((1 - s) * (xm[ia * 2 + 1] - p[ia * 2 + 1]) + s * (xm[ib * 2 + 1] - p[ib * 2 + 1]));
      const f64 tx = -ny, ty = nx, rt = rx * tx + ry * ty;
      const f64 hi = mu * pen, lo = -mu * pen;
      f64 ft = hi < rt ? hi : rt;
      ft = ft > lo ? ft : lo;
      const f64 lt = ft / den;
      xm[v * 2] -= w[v] * lt * tx; xm[v * 2 + 1] -= w[v] * lt * ty;
      xm[ia * 2] += wa * (1 - s) * lt * tx; xm[ia * 2 + 1] += wa * (1 - s) * lt * ty;
      xm[ib * 2] += wb * s * lt * tx; xm[ib * 2 + 1] += wb * s * lt * ty;
      contacts++;
      break;
    }
  }
  Q[Q_CONTACTS] = contacts;
}

// ================= главный шаг =================
extern "C" {
EXPORT("step") void step(f64 dt) {
  load_ctx();
  const i32 n = C.n, m = C.m, substeps = Q[Q_SUBSTEPS];
  const bool fem = Q[Q_FEM] != 0;
  f64 *x = C.x, *p = C.p, *v = C.v; const f64* w = C.w;
  const f64 h = dt / substeps, alpha = P[P_COMPL] / (h * h);
  const f64 maxX = P[P_BW] - 2, maxY = P[P_BH] - 2;
  const f64 grav = P[P_GRAV], damp = P[P_DAMP];
  Q[Q_CONTACTS] = 0;
#ifdef USE_SIMD
  if (fem && Q[Q_NCOLORS] > 0) prepareColored();
#endif

  for (i32 s = 0; s < substeps; s++) {
    // 1. предсказание позиций
    for (i32 i = 0; i < n; i++) {
      const i32 ix = i * 2, iy = ix + 1;
      p[ix] = x[ix]; p[iy] = x[iy];
      if (w[i] > 0) {
        v[iy] += grav * h;
        x[ix] += v[ix] * h; x[iy] += v[iy] * h;
      }
    }
    // 2. ограничения расстояния (только без FEM: режимы M1/M2)
    if (!fem) {
      for (i32 e = 0; e < m; e++) {
        if (!C.alive[e]) continue;
        const i32 a = C.edges[e * 2], b = C.edges[e * 2 + 1];
        const f64 wa = w[a], wb = w[b], ws = wa + wb;
        if (ws == 0) continue;
        f64 dx = x[a * 2] - x[b * 2], dy = x[a * 2 + 1] - x[b * 2 + 1];
        f64 len = hyp(dx, dy);
        if (len == 0) len = 1e-9;
        const f64 Cd = len - C.rest[e];
        const f64 dl = -Cd / (ws + alpha);
        dx /= len; dy /= len;
        x[a * 2] += wa * dl * dx; x[a * 2 + 1] += wa * dl * dy;
        x[b * 2] -= wb * dl * dx; x[b * 2 + 1] -= wb * dl * dy;
      }
    }
    // 2б. FEM (M3) или площадь (M2)
    if (fem) {
#ifdef USE_SIMD
      if (Q[Q_NCOLORS] > 0) femSolveColored(h); else
#endif
      femSolveScalar(h);
    } else if (Q[Q_AREA]) {
      const f64 aAlpha = P[P_ACOMPL] / (h * h);
      for (i32 t = 0; t < C.T; t++) {
        if (!C.alive[C.triEdges[t * 3]] || !C.alive[C.triEdges[t * 3 + 1]] || !C.alive[C.triEdges[t * 3 + 2]]) continue;
        const i32 i0 = C.tris[t * 3], i1 = C.tris[t * 3 + 1], i2 = C.tris[t * 3 + 2];
        const f64 x0 = x[i0 * 2], y0 = x[i0 * 2 + 1], x1 = x[i1 * 2], y1 = x[i1 * 2 + 1], x2 = x[i2 * 2], y2 = x[i2 * 2 + 1];
        const f64 Cd = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) - C.restArea[t];
        const f64 g0x = 0.5 * (y1 - y2), g0y = 0.5 * (x2 - x1);
        const f64 g1x = 0.5 * (y2 - y0), g1y = 0.5 * (x0 - x2);
        const f64 g2x = 0.5 * (y0 - y1), g2y = 0.5 * (x1 - x0);
        const f64 w0 = w[i0], w1 = w[i1], w2 = w[i2];
        const f64 den = w0 * (g0x * g0x + g0y * g0y) + w1 * (g1x * g1x + g1y * g1y) + w2 * (g2x * g2x + g2y * g2y) + aAlpha;
        if (den < 1e-12) continue;
        const f64 dl = -Cd / den;
        x[i0 * 2] += w0 * dl * g0x; x[i0 * 2 + 1] += w0 * dl * g0y;
        x[i1 * 2] += w1 * dl * g1x; x[i1 * 2 + 1] += w1 * dl * g1y;
        x[i2 * 2] += w2 * dl * g2x; x[i2 * 2 + 1] += w2 * dl * g2y;
      }
    }
    // 2в. столкновения между кусками (M5)
    if (fem && Q[Q_COLLIDE] && Q[Q_PIECES] > 1) collide();
    // 3. границы окна
    for (i32 i = 0; i < n; i++) {
      if (w[i] == 0) continue;
      const i32 ix = i * 2, iy = ix + 1;
      if (x[iy] > maxY) {
        x[iy] = maxY;
        x[ix] = p[ix] + (x[ix] - p[ix]) * 0.7; // трение об пол
      }
      if (x[ix] < 2) x[ix] = 2; else if (x[ix] > maxX) x[ix] = maxX;
    }
    // 4. скорости из изменения позиций
    for (i32 i = 0; i < n; i++) {
      if (w[i] == 0) { v[i * 2] = v[i * 2 + 1] = 0; continue; }
      v[i * 2] = (x[i * 2] - p[i * 2]) / h * damp;
      v[i * 2 + 1] = (x[i * 2 + 1] - p[i * 2 + 1]) / h * damp;
    }
  }

  {
    f64 m2 = 0;
    for (i32 i = 0; i < n; i++) { const f64 q = v[i * 2] * v[i * 2] + v[i * 2 + 1] * v[i * 2 + 1]; if (q > m2) m2 = q; }
    P[P_VMAX] = __builtin_sqrt(m2);
  }
  // деформация рёбер (раскраска / разрыв без FEM) — раз за кадр
  const bool tear = Q[Q_TEAR] != 0;
  for (i32 e = 0; e < m; e++) {
    if (!C.alive[e]) { C.strain[e] = 0; continue; }
    const i32 a = C.edges[e * 2], b = C.edges[e * 2 + 1];
    const f64 len = hyp(x[a * 2] - x[b * 2], x[a * 2 + 1] - x[b * 2 + 1]);
    const f64 st = __builtin_fabs(len - C.rest[e]) / C.rest[e];
    C.strain[e] = (f32)st;
    if (!fem && tear && len > C.rest[e] && st > P[P_TEARSTRAIN]) C.alive[e] = 0;
  }
  if (fem) updateStress();
}
}
