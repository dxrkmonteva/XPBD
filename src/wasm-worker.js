// Рабочий поток для многопоточного ядра: подключается к общей памяти и входит в worker_loop (см. wasm/core.cpp).
// Работает и в браузере (Worker), и в Node (worker_threads): окружение передаёт функцию post-ready.
export async function workerMain({ module, memory, tid, stackTop }, ready) {
  const inst = await WebAssembly.instantiate(module, { env: { memory, now: () => 0 } });
  inst.exports.__stack_pointer.value = stackTop; // собственный стек потока
  ready();
  inst.exports.worker_loop(tid); // не возвращается, пока главный поток не вызовет stop_workers
}
