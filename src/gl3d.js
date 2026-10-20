// M8: минимальный WebGL2-рендер без библиотек: поверхность тела (гладкие нормали, раскраска по деформации),
// пол с сеткой, проекционная тень (через stencil, чтобы перекрытия не темнели дважды) и орбитальная камера.

const M = {
  persp(fovy, asp, n, f) { const t = 1 / Math.tan(fovy / 2), o = new Float32Array(16); o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o; },
  look(e, c, u) {
    let zx = e[0] - c[0], zy = e[1] - c[1], zz = e[2] - c[2], l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
    let xx = u[1] * zz - u[2] * zy, xy = u[2] * zx - u[0] * zz, xz = u[0] * zy - u[1] * zx; l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx, o = new Float32Array(16);
    o[0] = xx; o[1] = yx; o[2] = zx; o[4] = xy; o[5] = yy; o[6] = zy; o[8] = xz; o[9] = yz; o[10] = zz;
    o[12] = -(xx * e[0] + xy * e[1] + xz * e[2]); o[13] = -(yx * e[0] + yy * e[1] + yz * e[2]); o[14] = -(zx * e[0] + zy * e[1] + zz * e[2]); o[15] = 1; return o;
  },
  mul(a, b) { const o = new Float32Array(16); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; } return o; },
};

export class Camera {
  constructor() { this.target = [1, 0.4, 0.25]; this.yaw = 0.7; this.pitch = 0.35; this.dist = 3.6; this.fov = 0.8; this.aspect = 1; }
  eye() {
    const cp = Math.cos(this.pitch), t = this.target, d = this.dist * Math.max(1, 0.9 / this.aspect); // узкий экран → дальше
    return [t[0] + d * cp * Math.sin(this.yaw), t[1] + d * Math.sin(this.pitch), t[2] + d * cp * Math.cos(this.yaw)];
  }
  matrices(aspect) {
    this.aspect = aspect;
    const view = M.look(this.eye(), this.target, [0, 1, 0]), proj = M.persp(this.fov, aspect, 0.05, 100);
    return { view, proj, vp: M.mul(proj, view) };
  }
  // луч через точку экрана (nx, ny ∈ [−1, 1], y вверх)
  ray(nx, ny, aspect) {
    this.aspect = aspect;
    const e = this.eye(), t = this.target;
    let fx = t[0] - e[0], fy = t[1] - e[1], fz = t[2] - e[2], l = Math.hypot(fx, fy, fz); fx /= l; fy /= l; fz /= l;
    let rx = fy * 0 - fz * 1, ry = fz * 0 - fx * 0, rz = fx * 1 - fy * 0; // f × up(0,1,0) = (−fz, 0, fx)
    rx = -fz; ry = 0; rz = fx; l = Math.hypot(rx, rz); rx /= l; rz /= l;
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx; // r × f
    const th = Math.tan(this.fov / 2), dx = fx + (rx * nx * aspect + ux * ny) * th, dy = fy + (ry * nx * aspect + uy * ny) * th, dz = fz + (rz * nx * aspect + uz * ny) * th;
    l = Math.hypot(dx, dy, dz);
    return { o: e, d: [dx / l, dy / l, dz / l], f: [fx, fy, fz] };
  }
}

const VS = `#version 300 es
in vec3 aPos; in vec3 aNrm; in float aDef;
uniform mat4 uVP; uniform float uShadow; uniform vec3 uLight;
out vec3 vN; out float vD; out vec3 vW;
void main() {
  vec3 p = aPos;
  if (uShadow > 0.5) { // проекция на пол вдоль направления света
    float t = p.y / uLight.y; p = vec3(p.x - uLight.x * t, 0.002, p.z - uLight.z * t);
  }
  vN = aNrm; vD = aDef; vW = aPos;
  gl_Position = uVP * vec4(p, 1.0);
}`;
const FS = `#version 300 es
precision highp float;
in vec3 vN; in float vD; in vec3 vW;
uniform float uShadow; uniform vec3 uLight; uniform float uColorize;
out vec4 o;
void main() {
  if (uShadow > 0.5) { o = vec4(0.0, 0.0, 0.0, 0.38); return; }
  vec3 n = normalize(vN);
  if (!gl_FrontFacing) n = -n;
  float dif = max(dot(n, normalize(uLight)), 0.0), hemi = 0.5 + 0.5 * n.y;
  float s = clamp(vD * uColorize, 0.0, 1.0);
  vec3 base = mix(vec3(0.30, 0.62, 0.95), vec3(1.0, 0.35, 0.18), s);
  o = vec4(base * (0.28 + 0.34 * hemi + 0.62 * dif), 1.0);
}`;
const VS_F = `#version 300 es
in vec2 aXZ; uniform mat4 uVP; out vec2 vXZ;
void main() { vXZ = aXZ; gl_Position = uVP * vec4(aXZ.x, 0.0, aXZ.y, 1.0); }`;
const FS_F = `#version 300 es
precision highp float;
in vec2 vXZ; out vec4 o; uniform vec2 uC;
void main() {
  vec2 g = abs(fract(vXZ - 0.5) - 0.5) / fwidth(vXZ);
  float line = 1.0 - min(min(g.x, g.y), 1.0);
  float fade = 1.0 - smoothstep(6.0, 18.0, length(vXZ - uC));
  vec3 c = mix(vec3(0.075, 0.09, 0.12), vec3(0.22, 0.28, 0.36), line);
  o = vec4(c, fade);
}`;

function program(gl, vs, fs) {
  const mk = (t, s) => { const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh); if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh)); return sh; };
  const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

export class Renderer3D {
  constructor(canvas) {
    const gl = this.gl = canvas.getContext('webgl2', { antialias: true, stencil: true, alpha: false });
    if (!gl) throw new Error('WebGL2 недоступен');
    this.canvas = canvas;
    this.pm = program(gl, VS, FS); this.pf = program(gl, VS_F, FS_F);
    this.vao = gl.createVertexArray(); this.vaoF = gl.createVertexArray();
    this.bufP = gl.createBuffer(); this.bufN = gl.createBuffer(); this.bufD = gl.createBuffer(); this.bufI = gl.createBuffer(); this.bufF = gl.createBuffer();
    gl.bindVertexArray(this.vaoF);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufF);
    const L = 40; gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-L, -L, L, -L, L, L, -L, -L, L, L, -L, L]), gl.STATIC_DRAW);
    const aXZ = gl.getAttribLocation(this.pf, 'aXZ'); gl.enableVertexAttribArray(aXZ); gl.vertexAttribPointer(aXZ, 2, gl.FLOAT, false, 0, 0);
    this.light = [0.45, 1.0, 0.35];
    this.colorize = 1; // масштаб раскраски деформации (0 — выкл.)
  }

  // Сетка поверхности: уникальные вершины граней → компактные буферы
  setMesh(sim) {
    const gl = this.gl, f = sim.faces;
    this.map = new Int32Array(sim.n).fill(-1); const ids = [];
    for (let i = 0; i < f.length; i++) if (this.map[f[i]] < 0) { this.map[f[i]] = ids.length; ids.push(f[i]); }
    this.ids = Int32Array.from(ids); this.nv = ids.length;
    this.idx = new Uint32Array(f.length); for (let i = 0; i < f.length; i++) this.idx[i] = this.map[f[i]];
    this.pos = new Float32Array(this.nv * 3); this.nrm = new Float32Array(this.nv * 3); this.def = new Float32Array(this.nv);
    // для усреднения деформации по вершинам: список тетраэдров на вершину (CSR)
    const cnt = new Int32Array(this.nv + 1);
    for (let t = 0; t < sim.T; t++) for (let k = 0; k < 4; k++) { const m = this.map[sim.tets[t * 4 + k]]; if (m >= 0) cnt[m + 1]++; }
    for (let i = 0; i < this.nv; i++) cnt[i + 1] += cnt[i];
    this.vtStart = cnt; this.vtList = new Int32Array(cnt[this.nv]); const fill = cnt.slice(0, this.nv);
    for (let t = 0; t < sim.T; t++) for (let k = 0; k < 4; k++) { const m = this.map[sim.tets[t * 4 + k]]; if (m >= 0) this.vtList[fill[m]++] = t; }
    gl.bindVertexArray(this.vao);
    const attr = (buf, name, size, data) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      const loc = gl.getAttribLocation(this.pm, name); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    };
    attr(this.bufP, 'aPos', 3, this.pos); attr(this.bufN, 'aNrm', 3, this.nrm); attr(this.bufD, 'aDef', 1, this.def);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.bufI); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.idx, gl.STATIC_DRAW);
    this.ni = this.idx.length;
  }

  _update(sim) {
    const x = sim.x, pos = this.pos, nrm = this.nrm, ids = this.ids, idx = this.idx;
    for (let i = 0; i < this.nv; i++) { const s = ids[i] * 3; pos[i * 3] = x[s]; pos[i * 3 + 1] = x[s + 1]; pos[i * 3 + 2] = x[s + 2]; }
    nrm.fill(0);
    for (let k = 0; k < idx.length; k += 3) {
      const a = idx[k] * 3, b = idx[k + 1] * 3, c = idx[k + 2] * 3;
      const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2], vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx; // не нормируем: вес ∝ площади
      for (const q of [a, b, c]) { nrm[q] += nx; nrm[q + 1] += ny; nrm[q + 2] += nz; }
    }
    for (let i = 0; i < this.nv; i++) {
      const l = Math.hypot(nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]) || 1; nrm[i * 3] /= l; nrm[i * 3 + 1] /= l; nrm[i * 3 + 2] /= l;
      let s = 0; const a = this.vtStart[i], b = this.vtStart[i + 1];
      for (let k = a; k < b; k++) s += sim.defo[this.vtList[k]];
      this.def[i] = b > a ? s / (b - a) : 0;
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufP); gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufN); gl.bufferSubData(gl.ARRAY_BUFFER, 0, nrm);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufD); gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.def);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2), c = this.canvas;
    const w = Math.floor(c.clientWidth * dpr), h = Math.floor(c.clientHeight * dpr);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return w / h;
  }

  draw(sim, cam) {
    const gl = this.gl, aspect = this.resize();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.043, 0.055, 0.078, 1); gl.clearStencil(0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    const { vp } = cam.matrices(aspect);
    this._update(sim);
    // пол
    gl.useProgram(this.pf); gl.bindVertexArray(this.vaoF);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.pf, 'uVP'), false, vp);
    gl.uniform2f(gl.getUniformLocation(this.pf, 'uC'), cam.target[0], cam.target[2]);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // тело
    gl.useProgram(this.pm); gl.bindVertexArray(this.vao);
    const U = n => gl.getUniformLocation(this.pm, n);
    gl.uniformMatrix4fv(U('uVP'), false, vp); gl.uniform3fv(U('uLight'), this.light); gl.uniform1f(U('uColorize'), this.colorize);
    // тень: stencil пропускает каждый пиксель один раз; глубину не пишем
    gl.uniform1f(U('uShadow'), 1);
    gl.enable(gl.STENCIL_TEST); gl.stencilFunc(gl.EQUAL, 0, 0xff); gl.stencilOp(gl.KEEP, gl.KEEP, gl.INCR); gl.depthMask(false);
    gl.drawElements(gl.TRIANGLES, this.ni, gl.UNSIGNED_INT, 0);
    gl.disable(gl.STENCIL_TEST); gl.depthMask(true); gl.disable(gl.BLEND);
    gl.uniform1f(U('uShadow'), 0);
    gl.drawElements(gl.TRIANGLES, this.ni, gl.UNSIGNED_INT, 0);
  }

  // Луч ∩ поверхность: возвращает {t, vertex (ближайшая вершина попавшего треугольника), point} или null
  pick(sim, ray) {
    const x = sim.x, f = sim.faces, o = ray.o, d = ray.d; let best = Infinity, bi = -1;
    for (let k = 0; k < f.length; k += 3) {
      const a = f[k] * 3, b = f[k + 1] * 3, c = f[k + 2] * 3;
      const e1x = x[b] - x[a], e1y = x[b + 1] - x[a + 1], e1z = x[b + 2] - x[a + 2], e2x = x[c] - x[a], e2y = x[c + 1] - x[a + 1], e2z = x[c + 2] - x[a + 2];
      const px = d[1] * e2z - d[2] * e2y, py = d[2] * e2x - d[0] * e2z, pz = d[0] * e2y - d[1] * e2x, det = e1x * px + e1y * py + e1z * pz;
      if (Math.abs(det) < 1e-12) continue;
      const inv = 1 / det, tx = o[0] - x[a], ty = o[1] - x[a + 1], tz = o[2] - x[a + 2], u = (tx * px + ty * py + tz * pz) * inv;
      if (u < 0 || u > 1) continue;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x, v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
      if (v < 0 || u + v > 1) continue;
      const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (t > 0 && t < best) { best = t; bi = k; }
    }
    if (bi < 0) return null;
    const pt = [o[0] + d[0] * best, o[1] + d[1] * best, o[2] + d[2] * best];
    let bv = -1, bd = Infinity;
    for (let j = 0; j < 3; j++) { const v = f[bi + j], dx = x[v * 3] - pt[0], dy = x[v * 3 + 1] - pt[1], dz = x[v * 3 + 2] - pt[2], dd = dx * dx + dy * dy + dz * dz; if (dd < bd) { bd = dd; bv = v; } }
    return { t: best, vertex: bv, point: pt };
  }
}
