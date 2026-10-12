// Точка входа Web Worker (браузер)
import { workerMain } from './wasm-worker.js';
self.onmessage = e => workerMain(e.data, () => self.postMessage({ ready: true }));
