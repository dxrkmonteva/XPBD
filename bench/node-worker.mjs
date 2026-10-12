// Точка входа воркера для Node (worker_threads)
import { parentPort } from 'worker_threads';
import { workerMain } from '../src/wasm-worker.js';
parentPort.on('message', d => workerMain(d, () => parentPort.postMessage({ ready: true })));
