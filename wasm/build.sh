#!/usr/bin/env bash
# Сборка ядра C++ → WebAssembly без Emscripten: достаточно clang и lld (apt install clang lld).
# Результат: src/wasm/core.scalar.wasm (f64, тот же порядок вычислений, что в JS) и
#            src/wasm/core.simd.wasm   (WASM SIMD, f32x4 + раскраска элементов; M6.2)
set -euo pipefail
cd "$(dirname "$0")"
OUT=../src/wasm
mkdir -p "$OUT"
CLANG="${CLANG:-clang}"

COMMON=(--target=wasm32 -O3 -ffreestanding -nostdlib -fno-exceptions -fno-rtti -mbulk-memory
        -Wall -Wno-misleading-indentation
        -Wl,--no-entry -Wl,--export-memory -Wl,--export=__heap_base -Wl,-z,stack-size=65536)

echo "scalar..."
"$CLANG" "${COMMON[@]}" -o "$OUT/core.scalar.wasm" core.cpp
ls -l "$OUT/core.scalar.wasm"

if grep -q "USE_SIMD" core.cpp; then
  echo "simd..."
  "$CLANG" "${COMMON[@]}" -msimd128 -DUSE_SIMD -o "$OUT/core.simd.wasm" core.cpp
  ls -l "$OUT/core.simd.wasm"
fi

if grep -q "USE_THREADS" core.cpp; then
  echo "simd + threads..."
  # общая (shared) память импортируется из JS; у каждого потока свой стек (JS выставляет __stack_pointer)
  "$CLANG" --target=wasm32 -O3 -ffreestanding -nostdlib -fno-exceptions -fno-rtti -mbulk-memory -matomics -msimd128 -DUSE_SIMD -DUSE_THREADS \
    -Wall -Wno-misleading-indentation -Wl,--no-entry -Wl,--shared-memory -Wl,--import-memory -Wl,--max-memory=268435456 \
    -Wl,--export=__heap_base -Wl,--export=__stack_pointer -Wl,-z,stack-size=65536 -o "$OUT/core.simd.mt.wasm" core.cpp
  ls -l "$OUT/core.simd.mt.wasm"
fi

# M8: 3D-ядро (отдельный модуль): скалярное и SIMD
echo "3d scalar..."
"$CLANG" "${COMMON[@]}" -o "$OUT/core3d.scalar.wasm" core3d.cpp
echo "3d simd..."
"$CLANG" "${COMMON[@]}" -msimd128 -DUSE_SIMD -o "$OUT/core3d.simd.wasm" core3d.cpp
ls -l "$OUT"/core3d.*.wasm
