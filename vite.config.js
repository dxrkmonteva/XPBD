// Заголовки для кросс-доменной изоляции: нужны браузеру, чтобы дать странице SharedArrayBuffer (многопоточный WASM).
const isolation = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };
export default {
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
};
