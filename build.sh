#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building all WASM projects in subdirectories..."
echo "Optimizing for size with gzip compression..."

# Find all subdirectories with Cargo.toml
for dir in */; do
    if [[ -f "${dir}Cargo.toml" ]]; then
        echo ""
        echo "=== Building ${dir%/} ==="
        cd "$dir"

        # Build with wasm-pack (--release uses profile.release settings)
        # Enable unstable WebGPU APIs for projects that need them (e.g. ray_tracer)
        RUSTFLAGS="${RUSTFLAGS:-} --cfg=web_sys_unstable_apis" \
            wasm-pack build --target web --release

        # Get the package name from Cargo.toml
        PKG_NAME=$(grep -m1 '^name' Cargo.toml | sed 's/name = "\(.*\)"/\1/')
        WASM_FILE="pkg/${PKG_NAME}_bg.wasm"
        JS_FILE="pkg/${PKG_NAME}.js"

        if [[ -f "$WASM_FILE" && -f "$JS_FILE" ]]; then
            ORIG_SIZE=$(stat -c%s "$WASM_FILE" 2>/dev/null || stat -f%z "$WASM_FILE")

            # Base64 encode the WASM file
            WASM_B64=$(base64 -w0 "$WASM_FILE")

            # Read the JS glue code
            JS_CODE=$(cat "$JS_FILE")

            # Create self-contained HTML bundle
            BUNDLE_FILE="pkg/${PKG_NAME}_bundle.html"
            cat > "$BUNDLE_FILE" << HTMLEOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${PKG_NAME}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body { width: 100%; height: 100%; overflow: hidden; }
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            background: #1a1a2e;
            color: #eee;
            font-family: system-ui, sans-serif;
        }
    </style>
</head>
<body>
    <script type="module">
// Inline WASM as base64
const wasmBase64 = "${WASM_B64}";

// Decode base64 to ArrayBuffer
function base64ToArrayBuffer(base64) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

// Modified JS glue code with inline WASM loading
${JS_CODE}

// Override the default init to use our inline WASM
const initFn =
    (typeof __wbg_init === "function" && __wbg_init) ||
    (typeof init === "function" && init);

if (!initFn) {
    throw new Error("WASM init function not found (expected __wbg_init or init)");
}

async function initWithInlineWasm() {
    const wasmBuffer = base64ToArrayBuffer(wasmBase64);
    return initFn(wasmBuffer);
}

initWithInlineWasm().catch((err) => {
    console.error("WASM init failed:", err);
    if (!navigator.gpu && document.body.children.length === 0) {
        document.body.textContent = "WebGPU not available. Use Chrome/Edge over localhost or HTTPS.";
    }
});
    </script>
</body>
</html>
HTMLEOF

            BUNDLE_SIZE=$(stat -c%s "$BUNDLE_FILE" 2>/dev/null || stat -f%z "$BUNDLE_FILE")

            # Gzip the bundle
            gzip -9 -k -f "$BUNDLE_FILE"
            GZ_SIZE=$(stat -c%s "${BUNDLE_FILE}.gz" 2>/dev/null || stat -f%z "${BUNDLE_FILE}.gz")

            RATIO=$((GZ_SIZE * 100 / ORIG_SIZE))
            echo "  WASM: ${ORIG_SIZE} bytes"
            echo "  Bundle: ${BUNDLE_SIZE} bytes -> ${GZ_SIZE} bytes (gzipped)"
        fi

        echo "Built ${dir%/} successfully"
        cd "$SCRIPT_DIR"
    fi
done

echo ""
echo "All WASM projects built successfully!"
echo "Output: pkg/*_bundle.html (self-contained, upload this)"
echo "        pkg/*_bundle.html.gz (pre-compressed)"
