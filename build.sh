#!/bin/bash
# NanoFTS 构建脚本
# 用于解决 maturin 需要 README.md 在 nanofts 目录下的问题

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
NANOFTS_DIR="$PROJECT_ROOT/nanofts"

echo "=== NanoFTS Build Script ==="
echo "Project root: $PROJECT_ROOT"
echo "NanoFTS dir: $NANOFTS_DIR"

# 复制 README.md 到 nanofts 目录
if [ -f "$PROJECT_ROOT/README.md" ]; then
    echo "Copying README.md to nanofts directory..."
    cp "$PROJECT_ROOT/README.md" "$NANOFTS_DIR/README.md"
else
    echo "Warning: README.md not found in project root"
fi

# 切换到 nanofts 目录
cd "$NANOFTS_DIR"

# 解析命令行参数
BUILD_MODE="develop"
RELEASE_FLAG="--release"

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_MODE="build"
            shift
            ;;
        --develop)
            BUILD_MODE="develop"
            shift
            ;;
        --sdist)
            BUILD_MODE="sdist"
            shift
            ;;
        --debug)
            RELEASE_FLAG=""
            shift
            ;;
        --release)
            RELEASE_FLAG="--release"
            shift
            ;;
        --out)
            OUT_DIR="$2"
            shift 2
            ;;
        *)
            # 传递其他参数给 maturin
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# 执行 maturin 命令
echo "Running maturin $BUILD_MODE $RELEASE_FLAG ${EXTRA_ARGS[*]}"

if [ "$BUILD_MODE" = "sdist" ]; then
    if [ -n "$OUT_DIR" ]; then
        maturin sdist --out "$OUT_DIR" "${EXTRA_ARGS[@]}"
    else
        maturin sdist "${EXTRA_ARGS[@]}"
    fi
elif [ "$BUILD_MODE" = "build" ]; then
    if [ -n "$OUT_DIR" ]; then
        maturin build $RELEASE_FLAG --out "$OUT_DIR" "${EXTRA_ARGS[@]}"
    else
        maturin build $RELEASE_FLAG "${EXTRA_ARGS[@]}"
    fi
else
    maturin develop $RELEASE_FLAG "${EXTRA_ARGS[@]}"
fi

echo "=== Build completed successfully ==="

