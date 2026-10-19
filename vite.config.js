// Заголовки для кросс-доменной изоляции: нужны браузеру, чтобы дать странице SharedArrayBuffer (многопоточный WASM).
const isolation = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };
import { resolve } from 'node:path';
export default {
  build: { rollupOptions: { input: { main: resolve(import.meta.dirname, 'index.html'), view3d: resolve(import.meta.dirname, '3d.html') } } },
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
};
