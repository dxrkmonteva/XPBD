// XPBD 3D — вычислительное ядро на C++ (WebAssembly, без libc), M8.
// Повторяет src/sim3d.js операция в операцию: скалярная сборка даёт те же числа, что JS.
// -DUSE_SIMD — тетраэдры раскрашены (элементы одного цвета не делят вершин) и обрабатываются по 4 штуки в f32x4.
// Массивы лежат в линейной памяти WASM, JS видит их как TypedArray-представления (см. src/wasm-sim3d.js).
// Индексы ctx ОБЯЗАНЫ совпадать с PN/QN в src/wasm-sim3d.js.

typedef int i32;
typedef unsigned int u32;
typedef float f32;
typedef double f64;
#define EXPORT(name) __attribute__((export_name(name)))

enum { P_GRAV, P_YOUNG, P_NU, P_DAMP, P_FRIC, P_FLOORY, P_DCOMPL, P_DX, P_DY, P_DZ, P_COUNT = 16 };
enum { Q_N, Q_T, Q_SUBSTEPS, Q_DRAGV, Q_NCOLORS, Q_SIMD, Q_DIRTY, Q_TP,
       Q_X, Q_P, Q_V, Q_W, Q_TETS, Q_DMINV, Q_REST, Q_QUAT, Q_DEFO, Q_CORDER, Q_CSTART, Q_ORD, Q_SOA, Q_COUNT = 48 };
static f64 P[P_COUNT];
static i32 Q[Q_COUNT];
extern "C" {
EXPORT("ctx_f64") f64* ctx_f64() { return P; }
EXPORT("ctx_i32") i32* ctx_i32() { return Q; }
}
template <class T> static inline T* ptr(i32 idx) { return (T*)(unsigned long)(u32)Q[idx]; }

struct Ctx { i32 n, T; f64 *x, *p, *v, *w, *dmInv, *rest, *quat; i32* tets; f32* defo; };
static Ctx C;
static void load_ctx() {
  C.n = Q[Q_N]; C.T = Q[Q_T];
  C.x = ptr<f64>(Q_X); C.p = ptr<f64>(Q_P); C.v = ptr<f64>(Q_V); C.w = ptr<f64>(Q_W);
  C.dmInv = ptr<f64>(Q_DMINV); C.rest = ptr<f64>(Q_REST); C.quat = ptr<f64>(Q_QUAT);
  C.tets = ptr<i32>(Q_TETS); C.defo = ptr<f32>(Q_DEFO);
}

static inline f64 muOf() { return P[P_YOUNG] / (2 * (1 + P[P_NU])); }
static inline f64 lamOf() { return P[P_YOUNG] * P[P_NU] / ((1 + P[P_NU]) * (1 - 2 * P[P_NU])); }

// ---------------- скалярная часть (зеркало JS) ----------------
static f64 Fm[9], Rm[9], Gm[9], Hm[9];

static void defGrad(i32 t, f64* F) {
  const f64* x = C.x; const f64* M = C.dmInv + t * 9;
  const i32 a = C.tets[t * 4] * 3, b = C.tets[t * 4 + 1] * 3, c = C.tets[t * 4 + 2] * 3, d = C.tets[t * 4 + 3] * 3;
  const f64 d1x = x[b] - x[a], d1y = x[b + 1] - x[a + 1], d1z = x[b + 2] - x[a + 2];
  const f64 d2x = x[c] - x[a], d2y = x[c + 1] - x[a + 1], d2z = x[c + 2] - x[a + 2];
  const f64 d3x = x[d] - x[a], d3y = x[d + 1] - x[a + 1], d3z = x[d + 2] - x[a + 2];
  F[0] = d1x * M[0] + d2x * M[3] + d3x * M[6]; F[1] = d1x * M[1] + d2x * M[4] + d3x * M[7]; F[2] = d1x * M[2] + d2x * M[5] + d3x * M[8];
  F[3] = d1y * M[0] + d2y * M[3] + d3y * M[6]; F[4] = d1y * M[1] + d2y * M[4] + d3y * M[7]; F[5] = d1y * M[2] + d2y * M[5] + d3y * M[8];
  F[6] = d1z * M[0] + d2z * M[3] + d3z * M[6]; F[7] = d1z * M[1] + d2z * M[4] + d3z * M[7]; F[8] = d1z * M[2] + d2z * M[5] + d3z * M[8];
}

static inline void quatR(f64 qx, f64 qy, f64 qz, f64 qw, f64* R) {
  const f64 xx = qx * qx, yy = qy * qy, zz = qz * qz, xy = qx * qy, xz = qx * qz, yz = qy * qz, wx = qw * qx, wy = qw * qy, wz = qw * qz;
  R[0] = 1 - 2 * (yy + zz); R[1] = 2 * (xy - wz); R[2] = 2 * (xz + wy);
  R[3] = 2 * (xy + wz); R[4] = 1 - 2 * (xx + zz); R[5] = 2 * (yz - wx);
  R[6] = 2 * (xz - wy); R[7] = 2 * (yz + wx); R[8] = 1 - 2 * (xx + yy);
}

static void rotation(i32 t, const f64* F, f64* R, i32 iters) {
  f64* q = C.quat + t * 4;
  f64 qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  for (i32 it = 0; it < iters; it++) {
    quatR(qx, qy, qz, qw, R);
    f64 ox = 0, oy = 0, oz = 0, dot = 0;
    for (i32 c = 0; c < 3; c++) {
      const f64 rx = R[c], ry = R[3 + c], rz = R[6 + c], fx = F[c], fy = F[3 + c], fz = F[6 + c];
      ox += ry * fz - rz * fy; oy += rz * fx - rx * fz; oz += rx * fy - ry * fx; dot += rx * fx + ry * fy + rz * fz;
    }
    const f64 k = 1 / (__builtin_fabs(dot) + 1e-9); ox *= k; oy *= k; oz *= k;
    if (ox * ox + oy * oy + oz * oz < 1e-18) break;
    const f64 sx = ox * 0.5, sy = oy * 0.5, sz = oz * 0.5;
    const f64 nx = qx + sx * qw + sy * qz - sz * qy, ny = qy + sy * qw + sz * qx - sx * qz;
    const f64 nz = qz + sz * qw + sx * qy - sy * qx, nw = qw - sx * qx - sy * qy - sz * qz;
    const f64 l = 1 / __builtin_sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
    qx = nx * l; qy = ny * l; qz = nz * l; qw = nw * l;
  }
  q[0] = qx; q[1] = qy; q[2] = qz; q[3] = qw;
  quatR(qx, qy, qz, qw, R);
}

static void applyC(i32 t, const f64* G, f64 Cv, f64 alphaBar) {
  f64* x = C.x; const f64* w = C.w; const f64* M = C.dmInv + t * 9; f64* H = Hm;
  for (i32 r = 0; r < 3; r++) for (i32 c = 0; c < 3; c++) H[r * 3 + c] = G[r * 3] * M[c * 3] + G[r * 3 + 1] * M[c * 3 + 1] + G[r * 3 + 2] * M[c * 3 + 2];
  const f64 h0x = -(H[0] + H[1] + H[2]), h0y = -(H[3] + H[4] + H[5]), h0z = -(H[6] + H[7] + H[8]);
  const i32 p0 = C.tets[t * 4], p1 = C.tets[t * 4 + 1], p2 = C.tets[t * 4 + 2], p3 = C.tets[t * 4 + 3];
  const f64 w0 = w[p0], w1 = w[p1], w2 = w[p2], w3 = w[p3];
  const f64 den = w0 * (h0x * h0x + h0y * h0y + h0z * h0z) + w1 * (H[0] * H[0] + H[3] * H[3] + H[6] * H[6]) +
                  w2 * (H[1] * H[1] + H[4] * H[4] + H[7] * H[7]) + w3 * (H[2] * H[2] + H[5] * H[5] + H[8] * H[8]) + alphaBar;
  if (den < 1e-30) return;
  const f64 dl = -Cv / den;
  x[p0 * 3] += w0 * dl * h0x; x[p0 * 3 + 1] += w0 * dl * h0y; x[p0 * 3 + 2] += w0 * dl * h0z;
  x[p1 * 3] += w1 * dl * H[0]; x[p1 * 3 + 1] += w1 * dl * H[3]; x[p1 * 3 + 2] += w1 * dl * H[6];
  x[p2 * 3] += w2 * dl * H[1]; x[p2 * 3 + 1] += w2 * dl * H[4]; x[p2 * 3 + 2] += w2 * dl * H[7];
  x[p3 * 3] += w3 * dl * H[2]; x[p3 * 3 + 1] += w3 * dl * H[5]; x[p3 * 3 + 2] += w3 * dl * H[8];
}

static void femSolveScalar(f64 h) {
  const f64 mu = muOf(), lam = lamOf(), h2 = h * h;
  f64 *F = Fm, *R = Rm, *G = Gm;
  for (i32 t = 0; t < C.T; t++) {
    const f64 vol = C.rest[t];
    defGrad(t, F);
    rotation(t, F, R, 2);
    f64 s = 0;
    for (i32 i = 0; i < 9; i++) { const f64 e = F[i] - R[i]; G[i] = e; s += e * e; }
    const f64 Cd = __builtin_sqrt(s);
    C.defo[t] = (f32)Cd;
    if (Cd > 1e-9) { for (i32 i = 0; i < 9; i++) G[i] /= Cd; applyC(t, G, Cd, 1 / (2 * mu * vol) / h2); }
    defGrad(t, F);
    const f64 f0x = F[0], f0y = F[3], f0z = F[6], f1x = F[1], f1y = F[4], f1z = F[7], f2x = F[2], f2y = F[5], f2z = F[8];
    const f64 det = f0x * (f1y * f2z - f1z * f2y) - f1x * (f0y * f2z - f0z * f2y) + f2x * (f0y * f1z - f0z * f1y);
    if (lam > 0 && __builtin_fabs(det - 1) > 1e-12) {
      G[0] = f1y * f2z - f1z * f2y; G[3] = f1z * f2x - f1x * f2z; G[6] = f1x * f2y - f1y * f2x;
      G[1] = f2y * f0z - f2z * f0y; G[4] = f2z * f0x - f2x * f0z; G[7] = f2x * f0y - f2y * f0x;
      G[2] = f0y * f1z - f0z * f1y; G[5] = f0z * f1x - f0x * f1z; G[8] = f0x * f1y - f0y * f1x;
      applyC(t, G, det - 1, 1 / (lam * vol) / h2);
    }
  }
}

#ifdef USE_SIMD
#include <wasm_simd128.h>
// ---------------- SIMD: 4 тетраэдра одного цвета за раз (f32x4) ----------------
typedef float f4 __attribute__((vector_size(16)));
typedef int m4 __attribute__((vector_size(16)));
#define F4(x) ((f4){(x), (x), (x), (x)})
#define LD(p) ((f4)wasm_v128_load(p))
#define SQRT4(v) ((f4)wasm_f32x4_sqrt((v128_t)(v)))
#define ABS4(v) ((f4)wasm_f32x4_abs((v128_t)(v)))
static inline f4 sel(m4 m, f4 a, f4 b) { return (f4)wasm_v128_bitselect((v128_t)a, (v128_t)b, (v128_t)m); }

// SoA-данные тетраэдров в порядке раскраски (группами по 4; хвост цвета дополнен пустыми слотами)
enum { O_I0, O_I1, O_I2, O_I3, O_M0, O_M1, O_M2, O_M3, O_M4, O_M5, O_M6, O_M7, O_M8, O_IV, O_ACT, O_W0, O_W1, O_W2, O_W3,
       O_Q0, O_Q1, O_Q2, O_Q3, O_N };

static void prepareStatic() {
  const i32 Tp = Q[Q_TP]; char* base = ptr<char>(Q_SOA); const i32* order = ptr<i32>(Q_ORD);
  i32* oi = (i32*)base; f32* of = (f32*)base;
  for (i32 s = 0; s < Tp; s++) {
    const i32 t = order[s];
    if (t < 0) {
      oi[(O_I0) * Tp + s] = oi[(O_I1) * Tp + s] = oi[(O_I2) * Tp + s] = oi[(O_I3) * Tp + s] = 0;
      for (i32 k = 0; k < 9; k++) of[(O_M0 + k) * Tp + s] = (k % 4 == 0) ? 1.0f : 0.0f;
      of[O_IV * Tp + s] = 1.0f; of[O_ACT * Tp + s] = 0.0f;
      of[O_Q0 * Tp + s] = 0; of[O_Q1 * Tp + s] = 0; of[O_Q2 * Tp + s] = 0; of[O_Q3 * Tp + s] = 1;
      continue;
    }
    for (i32 k = 0; k < 4; k++) oi[(O_I0 + k) * Tp + s] = C.tets[t * 4 + k] * 3;
    for (i32 k = 0; k < 9; k++) of[(O_M0 + k) * Tp + s] = (f32)C.dmInv[t * 9 + k];
    of[O_IV * Tp + s] = (f32)(1.0 / C.rest[t]); of[O_ACT * Tp + s] = 1.0f;
    // стартовый поворот: уточняем скалярно (в f64) с запасом итераций
    f64 F[9], R[9]; defGrad(t, F); rotation(t, F, R, 20);
    for (i32 k = 0; k < 4; k++) of[(O_Q0 + k) * Tp + s] = (f32)C.quat[t * 4 + k];
  }
}

static void prepareDynamic() { // обратные массы вершин: меняются при закреплении
  const i32 Tp = Q[Q_TP]; char* base = ptr<char>(Q_SOA); const i32* oi = (const i32*)base; f32* of = (f32*)base; const f64* w = C.w;
  for (i32 s = 0; s < Tp; s++) for (i32 k = 0; k < 4; k++) of[(O_W0 + k) * Tp + s] = (f32)w[oi[(O_I0 + k) * Tp + s] / 3];
}

struct V3 { f4 x, y, z; };
static inline void quatR4(f4 qx, f4 qy, f4 qz, f4 qw, f4* R) {
  const f4 xx = qx * qx, yy = qy * qy, zz = qz * qz, xy = qx * qy, xz = qx * qz, yz = qy * qz, wx = qw * qx, wy = qw * qy, wz = qw * qz, one = F4(1.0f), two = F4(2.0f);
  R[0] = one - two * (yy + zz); R[1] = two * (xy - wz); R[2] = two * (xz + wy);
  R[3] = two * (xy + wz); R[4] = one - two * (xx + zz); R[5] = two * (yz - wx);
  R[6] = two * (xz - wy); R[7] = two * (yz + wx); R[8] = one - two * (xx + yy);
}

// Поправки к 4 вершинам для ограничения с градиентом G (9 f4) — как applyC, но без записи
struct Del { f4 x[4], y[4], z[4]; };
static inline void deltaC(const f4* M, const f4* w, const f4* G, f4 Cv, f4 alphaBar, m4 active, Del& d) {
  f4 H[9];
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) H[r * 3 + c] = G[r * 3] * M[c * 3] + G[r * 3 + 1] * M[c * 3 + 1] + G[r * 3 + 2] * M[c * 3 + 2];
  const f4 hx[4] = {-(H[0] + H[1] + H[2]), H[0], H[1], H[2]}, hy[4] = {-(H[3] + H[4] + H[5]), H[3], H[4], H[5]}, hz[4] = {-(H[6] + H[7] + H[8]), H[6], H[7], H[8]};
  f4 den = alphaBar;
  for (int i = 0; i < 4; i++) den = den + w[i] * (hx[i] * hx[i] + hy[i] * hy[i] + hz[i] * hz[i]);
  const m4 ok = active & (den > F4(1e-30f));
  const f4 dl = sel(ok, -Cv / sel(ok, den, F4(1.0f)), F4(0.0f));
  for (int i = 0; i < 4; i++) { d.x[i] = w[i] * dl * hx[i]; d.y[i] = w[i] * dl * hy[i]; d.z[i] = w[i] * dl * hz[i]; }
}

static void femColor(i32 col, f64 h) {
  const i32 Tp = Q[Q_TP]; char* base = ptr<char>(Q_SOA); const i32* cstart = ptr<i32>(Q_CSTART); const i32* order = ptr<i32>(Q_ORD);
  i32* oi = (i32*)base; f32* of = (f32*)base; f64* x = C.x;
  const f32 mu = (f32)muOf(), lam = (f32)lamOf(), ih2 = (f32)(1.0 / (h * h));
  const f4 aD0 = F4(0.5f / mu * ih2), aH0 = F4(lam > 0 ? 1.0f / lam * ih2 : 0.0f);
  const m4 lamOn = (m4){lam > 0, lam > 0, lam > 0, lam > 0} != (m4){0, 0, 0, 0};
  for (i32 s = cstart[col]; s < cstart[col + 1]; s += 4) {
    const i32* I[4] = {oi + O_I0 * Tp + s, oi + O_I1 * Tp + s, oi + O_I2 * Tp + s, oi + O_I3 * Tp + s};
    // gather: векторы рёбер d1,d2,d3 (в f64 → f32)
    f32 bd[9][4];
    for (int l = 0; l < 4; l++) {
      const f64 *a = x + I[0][l], *b = x + I[1][l], *c = x + I[2][l], *d = x + I[3][l];
      bd[0][l] = (f32)(b[0] - a[0]); bd[1][l] = (f32)(b[1] - a[1]); bd[2][l] = (f32)(b[2] - a[2]);
      bd[3][l] = (f32)(c[0] - a[0]); bd[4][l] = (f32)(c[1] - a[1]); bd[5][l] = (f32)(c[2] - a[2]);
      bd[6][l] = (f32)(d[0] - a[0]); bd[7][l] = (f32)(d[1] - a[1]); bd[8][l] = (f32)(d[2] - a[2]);
    }
    f4 dv[9]; for (int k = 0; k < 9; k++) dv[k] = LD(bd[k]);
    f4 M[9]; for (int k = 0; k < 9; k++) M[k] = LD(of + (O_M0 + k) * Tp + s);
    const f4 IV = LD(of + O_IV * Tp + s);
    const m4 act = LD(of + O_ACT * Tp + s) > F4(0.5f);
    f4 w[4]; for (int k = 0; k < 4; k++) w[k] = LD(of + (O_W0 + k) * Tp + s);
    f4 qx = LD(of + O_Q0 * Tp + s), qy = LD(of + O_Q1 * Tp + s), qz = LD(of + O_Q2 * Tp + s), qw = LD(of + O_Q3 * Tp + s);

    // F[r][c] = d1[r]*M[0][c] + d2[r]*M[1][c] + d3[r]*M[2][c]; dv: d1 = 0..2, d2 = 3..5, d3 = 6..8 (x,y,z)
    f4 F[9];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) F[r * 3 + c] = dv[r] * M[c] + dv[3 + r] * M[3 + c] + dv[6 + r] * M[6 + c];

    // ближайший поворот: 2 итерации по кватерниону (тёплый старт)
    f4 R[9];
    for (int it = 0; it < 2; it++) {
      quatR4(qx, qy, qz, qw, R);
      f4 ox = F4(0.0f), oy = F4(0.0f), oz = F4(0.0f), dot = F4(0.0f);
      for (int c = 0; c < 3; c++) {
        const f4 rx = R[c], ry = R[3 + c], rz = R[6 + c], fx = F[c], fy = F[3 + c], fz = F[6 + c];
        ox = ox + (ry * fz - rz * fy); oy = oy + (rz * fx - rx * fz); oz = oz + (rx * fy - ry * fx); dot = dot + (rx * fx + ry * fy + rz * fz);
      }
      const f4 k = F4(1.0f) / (ABS4(dot) + F4(1e-9f)); ox = ox * k; oy = oy * k; oz = oz * k;
      const f4 sx = ox * F4(0.5f), sy = oy * F4(0.5f), sz = oz * F4(0.5f);
      const f4 nx = qx + sx * qw + sy * qz - sz * qy, ny = qy + sy * qw + sz * qx - sx * qz;
      const f4 nz = qz + sz * qw + sx * qy - sy * qx, nw = qw - sx * qx - sy * qy - sz * qz;
      const f4 l = F4(1.0f) / SQRT4(nx * nx + ny * ny + nz * nz + nw * nw);
      qx = nx * l; qy = ny * l; qz = nz * l; qw = nw * l;
    }
    quatR4(qx, qy, qz, qw, R);
    { f32 t0[4], t1[4], t2[4], t3[4]; wasm_v128_store(t0, (v128_t)qx); wasm_v128_store(t1, (v128_t)qy); wasm_v128_store(t2, (v128_t)qz); wasm_v128_store(t3, (v128_t)qw);
      for (int l = 0; l < 4; l++) { of[O_Q0 * Tp + s + l] = t0[l]; of[O_Q1 * Tp + s + l] = t1[l]; of[O_Q2 * Tp + s + l] = t2[l]; of[O_Q3 * Tp + s + l] = t3[l]; } }

    // 1) девиаторная часть: C = ‖F − R‖, G = (F − R)/C
    f4 E[9]; f4 s2 = F4(0.0f);
    for (int k = 0; k < 9; k++) { E[k] = F[k] - R[k]; s2 = s2 + E[k] * E[k]; }
    const f4 Cd = SQRT4(s2);
    const m4 okd = act & (Cd > F4(1e-9f));
    const f4 invC = F4(1.0f) / sel(okd, Cd, F4(1.0f));
    f4 G[9]; for (int k = 0; k < 9; k++) G[k] = E[k] * invC;
    Del d1; deltaC(M, w, G, Cd, aD0 * IV * F4(1.0f), okd, d1);
    { f32 cd[4]; wasm_v128_store(cd, (v128_t)Cd); for (int l = 0; l < 4; l++) { const i32 t = order[s + l]; if (t >= 0) C.defo[t] = cd[l]; } }

    // позиции после девиатора: d_c += δ_{c+1} − δ_0
    f4 dv2[9];
    for (int c = 0; c < 3; c++) { dv2[c * 3] = dv[c * 3] + (d1.x[c + 1] - d1.x[0]); dv2[c * 3 + 1] = dv[c * 3 + 1] + (d1.y[c + 1] - d1.y[0]); dv2[c * 3 + 2] = dv[c * 3 + 2] + (d1.z[c + 1] - d1.z[0]); }
    f4 F2[9];
    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) F2[r * 3 + c] = dv2[r] * M[c] + dv2[3 + r] * M[3 + c] + dv2[6 + r] * M[6 + c];
    // 2) объёмная часть: C = det F − 1, G = cof(F)
    const f4 f0x = F2[0], f0y = F2[3], f0z = F2[6], f1x = F2[1], f1y = F2[4], f1z = F2[7], f2x = F2[2], f2y = F2[5], f2z = F2[8];
    const f4 det = f0x * (f1y * f2z - f1z * f2y) - f1x * (f0y * f2z - f0z * f2y) + f2x * (f0y * f1z - f0z * f1y);
    const m4 okv = act & (ABS4(det - F4(1.0f)) > F4(1e-12f)) & lamOn;
    f4 G2[9];
    G2[0] = f1y * f2z - f1z * f2y; G2[3] = f1z * f2x - f1x * f2z; G2[6] = f1x * f2y - f1y * f2x;
    G2[1] = f2y * f0z - f2z * f0y; G2[4] = f2z * f0x - f2x * f0z; G2[7] = f2x * f0y - f2y * f0x;
    G2[2] = f0y * f1z - f0z * f1y; G2[5] = f0z * f1x - f0x * f1z; G2[8] = f0x * f1y - f0y * f1x;
    Del d2; deltaC(M, w, G2, det - F4(1.0f), aH0 * IV, okv, d2);

    // единственный scatter за группу
    f32 sx_[4][4], sy_[4][4], sz_[4][4];
    for (int i = 0; i < 4; i++) {
      const f4 ax = d1.x[i] + d2.x[i], ay = d1.y[i] + d2.y[i], az = d1.z[i] + d2.z[i];
      wasm_v128_store(sx_[i], (v128_t)ax); wasm_v128_store(sy_[i], (v128_t)ay); wasm_v128_store(sz_[i], (v128_t)az);
    }
    for (int l = 0; l < 4; l++) for (int i = 0; i < 4; i++) { f64* q = x + I[i][l]; q[0] += sx_[i][l]; q[1] += sy_[i][l]; q[2] += sz_[i][l]; }
  }
}

static void femSolveColored(f64 h) { for (i32 col = 0; col < Q[Q_NCOLORS]; col++) femColor(col, h); }
#endif

// ---------------- шаг ----------------
extern "C" EXPORT("step3") void step3(f64 dt) {
  load_ctx();
  const i32 n = C.n, S = Q[Q_SUBSTEPS]; const f64 h = dt / S;
  f64 *x = C.x, *p = C.p, *v = C.v; const f64* w = C.w;
#ifdef USE_SIMD
  const bool simd = Q[Q_SIMD] && Q[Q_NCOLORS] > 0;
  if (simd) { if (Q[Q_DIRTY]) { prepareStatic(); Q[Q_DIRTY] = 0; } prepareDynamic(); }
#endif
  const i32 dv = Q[Q_DRAGV];
  for (i32 s = 0; s < S; s++) {
    for (i32 i = 0; i < n; i++) {
      const i32 a = i * 3;
      p[a] = x[a]; p[a + 1] = x[a + 1]; p[a + 2] = x[a + 2];
      if (w[i] > 0) { v[a + 1] -= P[P_GRAV] * h; x[a] += v[a] * h; x[a + 1] += v[a + 1] * h; x[a + 2] += v[a + 2] * h; }
    }
#ifdef USE_SIMD
    if (simd) femSolveColored(h); else
#endif
    femSolveScalar(h);
    if (dv >= 0 && w[dv] > 0) {
      const i32 a = dv * 3; const f64 dx = x[a] - P[P_DX], dy = x[a + 1] - P[P_DY], dz = x[a + 2] - P[P_DZ], L = __builtin_sqrt(dx * dx + dy * dy + dz * dz);
      if (L > 1e-9) { const f64 dl = -L / (w[dv] + P[P_DCOMPL] / (h * h)); x[a] += w[dv] * dl * dx / L; x[a + 1] += w[dv] * dl * dy / L; x[a + 2] += w[dv] * dl * dz / L; }
    }
    const f64 fy = P[P_FLOORY], fr = 1 - P[P_FRIC];
    for (i32 i = 0; i < n; i++) {
      if (w[i] == 0) continue;
      const i32 a = i * 3;
      if (x[a + 1] < fy) { x[a + 1] = fy; x[a] = p[a] + (x[a] - p[a]) * fr; x[a + 2] = p[a + 2] + (x[a + 2] - p[a + 2]) * fr; }
    }
    const f64 k = P[P_DAMP] / h;
    for (i32 i = 0; i < n * 3; i++) v[i] = (x[i] - p[i]) * k;
    for (i32 i = 0; i < n; i++) if (w[i] == 0) v[i * 3] = v[i * 3 + 1] = v[i * 3 + 2] = 0;
  }
}
