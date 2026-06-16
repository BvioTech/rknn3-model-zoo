#!/usr/bin/env bash
# convert.sh — Qwen3 LLM 一键转换(PC 端,x86_64,无需目标板)
#
# 封装两步:
#   1) export_llm.py   HF 模型 → .onnx / .config.pkl / .tokenizer.gguf / .embed.bin
#   2) export_rknn.py  .onnx → .rknn / .weight  (GRQ w4a16 group32,默认 --platform rk1828)
#
# 产物落在 <outdir>(默认 examples/Qwen3/model/llm/),即部署到板子的全部文件。
# 详见 docs/rk1828/01-模型转换.md。
set -euo pipefail

# ---- 路径定位(脚本在 examples/Qwen3/,仓库根在上两级) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY_DIR="$SCRIPT_DIR/python"
PY="$REPO_ROOT/.venv/bin/python"

# ---- 默认参数 ----
MODEL="Qwen/Qwen3-0.6B"
PLATFORM="rk1828"
OUTDIR="$REPO_ROOT/examples/Qwen3/model/llm"
NAME=""                 # 留空 = 取 MODEL 的 basename
QUANT=""                # --quant 透传(AWQ+GRQ 预量化,需 CUDA)
MODELSCOPE=""           # --modelscope 透传(从 modelscope 下载)
SKIP_ONNX=0             # 跳过 export_llm(复用已有 onnx/config/数据集)
SKIP_RKNN=0             # 只做 export_llm
PROXY="${https_proxy:-}"   # HF 下载代理,默认沿用环境
HF_ENDPOINT_ARG=""      # HF 镜像(不带结尾斜杠)

usage() {
    cat <<'EOF'
用法: convert.sh [选项] [<model>]

  <model>            HF 模型 id 或本地路径 (默认 Qwen/Qwen3-0.6B)

选项:
  -p, --platform P   目标平台 (默认 rk1828;RK1828 必须用 rk1828,勿用默认 rk1820)
  -o, --outdir DIR   产物输出目录 (默认 examples/Qwen3/model/llm)
  -n, --name NAME    产物基名 (默认取 model 的 basename,如 Qwen3-0.6B)
      --quant        启用 AWQ+GRQ 预量化 (需 CUDA;无 GPU 时底层自动跳过)
      --modelscope   从 modelscope.cn 下载模型(国内更快)
      --proxy URL    HF 下载代理 (设置 http_proxy/https_proxy,如 http://localhost:7897)
      --hf-endpoint URL  HF 镜像 (如 https://hf-mirror.com,不带结尾斜杠)
      --skip-onnx    跳过 export_llm,复用已有 .onnx/.config/数据集,只做 RKNN 转换
      --skip-rknn    只做 export_llm,不转 RKNN
  -h, --help         显示本帮助

示例:
  ./convert.sh Qwen/Qwen3-0.6B
  ./convert.sh --proxy http://localhost:7897 Qwen/Qwen3-1.7B
  ./convert.sh --modelscope Qwen/Qwen3-4B -p rk1828
  ./convert.sh --skip-onnx Qwen/Qwen3-0.6B      # 仅重跑量化(复用 onnx)
EOF
}

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform)   PLATFORM="$2"; shift 2 ;;
        -o|--outdir)     OUTDIR="$2"; shift 2 ;;
        -n|--name)       NAME="$2"; shift 2 ;;
        --quant)         QUANT="--quant"; shift ;;
        --modelscope)    MODELSCOPE="--modelscope"; shift ;;
        --proxy)         PROXY="$2"; shift 2 ;;
        --hf-endpoint)   HF_ENDPOINT_ARG="$2"; shift 2 ;;
        --skip-onnx)     SKIP_ONNX=1; shift ;;
        --skip-rknn)     SKIP_RKNN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        -*)              echo "未知选项: $1" >&2; usage; exit 2 ;;
        *)               MODEL="$1"; shift ;;
    esac
done

[[ -z "$NAME" ]] && NAME="$(basename "$MODEL")"

ONNX="$OUTDIR/$NAME.onnx"
CONFIG="$OUTDIR/$NAME.config.pkl"
RKNN="$OUTDIR/$NAME.rknn"

# ---- 前置检查 ----
[[ -x "$PY" ]] || { echo "✗ 找不到 venv: $PY (先按 docs/rk1828/01 建 .venv 并装 full toolkit)" >&2; exit 1; }
"$PY" -c "from rknn.api import RKNN" 2>/dev/null \
    || { echo "✗ venv 里没装 full toolkit (rknn.api 导入失败) — 勿用 lite wheel" >&2; exit 1; }

# ---- 环境 ----
export PYTHONPATH="$REPO_ROOT"
if [[ -n "$PROXY" ]]; then export http_proxy="$PROXY" https_proxy="$PROXY"; fi
if [[ -n "$HF_ENDPOINT_ARG" ]]; then export HF_ENDPOINT="$HF_ENDPOINT_ARG"; fi
mkdir -p "$OUTDIR"

echo "════════════════════════════════════════════════════════"
echo " 模型      : $MODEL"
echo " 平台      : $PLATFORM"
echo " 输出      : $OUTDIR/$NAME.{onnx,config.pkl,tokenizer.gguf,embed.bin,rknn,weight}"
echo " 代理      : ${PROXY:-<无>}    HF_ENDPOINT: ${HF_ENDPOINT_ARG:-<默认 hf.co>}"
echo "════════════════════════════════════════════════════════"

cd "$PY_DIR"

# ---- 步骤 1: 导出 ONNX + tokenizer + embed + 量化数据集 ----
if [[ "$SKIP_ONNX" -eq 0 ]]; then
    echo ""
    echo "[1/2] export_llm.py  →  ONNX / config / tokenizer / embed ..."
    t0=$SECONDS
    "$PY" export_llm.py \
        --model_path "$MODEL" \
        --export_llm_path "$ONNX" \
        $QUANT $MODELSCOPE
    echo "  ✓ export_llm 完成 ($((SECONDS - t0))s)"
else
    echo "[1/2] 跳过 export_llm (--skip-onnx);复用 $ONNX"
    [[ -f "$ONNX" ]]    || { echo "✗ 缺少 $ONNX" >&2; exit 1; }
fi

# ---- 步骤 2: 转 RKNN(GRQ 量化) ----
if [[ "$SKIP_RKNN" -eq 0 ]]; then
    echo ""
    echo "[2/2] export_rknn.py --platform $PLATFORM  →  .rknn / .weight ..."
    echo "      (GRQ w4a16 group32;无 CUDA 走 CPU,0.6B 约 15–20 分钟)"
    t0=$SECONDS
    "$PY" export_rknn.py \
        --platform "$PLATFORM" \
        --onnx_path "$ONNX" \
        --config "$CONFIG" \
        --rknn_path "$RKNN"
    echo "  ✓ export_rknn 完成 ($((SECONDS - t0))s)"
else
    echo "[2/2] 跳过 export_rknn (--skip-rknn)"
fi

# ---- 产物汇总 ----
echo ""
echo "════════════════════════════════════════════════════════"
echo " 产物 ($OUTDIR):"
for ext in onnx config.pkl tokenizer.gguf embed.bin rknn weight; do
    f="$OUTDIR/$NAME.$ext"
    if [[ -f "$f" ]]; then
        printf "   ✓ %-22s %s\n" "$NAME.$ext" "$(du -h "$f" | cut -f1)"
    else
        printf "   · %-22s (未生成)\n" "$NAME.$ext"
    fi
done
echo "════════════════════════════════════════════════════════"
echo " 部署到板子需要: .rknn  .weight  .embed.bin  + tokenizer (见 docs/rk1828/03)"
