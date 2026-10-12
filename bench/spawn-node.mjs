import { Worker } from 'worker_threads';
// Обёртка воркера Node под интерфейс wasm-sim.js: post / onready / onerror / terminate
export function spawnNode() {
  const w = new Worker(new URL('./node-worker.mjs', import.meta.url));
  return {
    post: d => w.postMessage(d),
    onready: cb => w.once('message', m => m.ready && cb()),
    onerror: cb => w.once('error', cb),
    terminate: () => w.terminate(),
  };
}
