#!/usr/bin/env bash
# convert.sh — BAAI/bge-reranker-v2-m3 一键转换(PC 端,x86_64,无需目标板)
#
#   1) export_onnx.py  HF 模型 → .onnx (+ .embed.bin + tokenizer/)   [固定 seq_len,默认 512]
#   2) export_rknn.py  .onnx → .rknn / .weight                       [默认 fp16,不量化]
#
# bge-reranker-v2-m3 是 XLM-RoBERTa encoder(输出单个相关性分数),与 Qwen3-Reranker
# (decoder-only,走 load_llm 流式)是两套不同路线;这里走标准 load_onnx+build。
# ⚠️ RK1828 固件约束:词向量查表(250002x1024 大表 Gather)放 host 端做 —— 模型图以
#    inputs_embeds 为输入,词表导出为 .embed.bin;position_ids 也由 host 预算后显式喂入
#    (去掉内部 CumSum)。否则板端 MODEL_SETUP fail。详见 README.md。
# 复用仓库根 .venv(full toolkit,torch 2.7 / transformers 4.51.3)。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY_DIR="$SCRIPT_DIR/python"
PY="${BGE_PY:-$REPO_ROOT/.venv/bin/python}"

MODEL="BAAI/bge-reranker-v2-m3"
PLATFORM="rk1828"
OUTDIR="$SCRIPT_DIR/model"
NAME="bge-reranker-v2-m3"
SEQ_LEN=512
QUANT=""
DATASET=""
MODELSCOPE=""
SKIP_ONNX=0
SKIP_RKNN=0
PROXY="${https_proxy:-}"
HF_ENDPOINT_ARG=""

usage() {
    cat <<'EOF'
用法: convert.sh [选项] [<model>]

  <model>            HF 模型 id 或本地路径 (默认 BAAI/bge-reranker-v2-m3)

选项:
  -p, --platform P   目标平台 (默认 rk1828)
  -o, --outdir DIR   产物输出目录 (默认 ./model)
  -n, --name NAME    产物基名 (默认 bge-reranker-v2-m3)
  -L, --seq-len N    固定序列长度 (默认 512;export_onnx/export_rknn 必须一致)
      --quant        w8a8 int8 量化 (需 --dataset;默认 fp16 不量化)
      --dataset F    量化校准列表 (.txt,每行 npy 路径)
      --modelscope   从 modelscope.cn 下载
      --proxy URL    HF 下载代理 (如 http://localhost:7897)
      --hf-endpoint URL  HF 镜像 (不带结尾斜杠)
      --py PATH      指定 python (默认 <repo>/.venv/bin/python 或 $BGE_PY)
      --skip-onnx    复用已有 onnx,只转 RKNN
      --skip-rknn    只导出 ONNX
  -h, --help

示例:
  ./convert.sh
  ./convert.sh --proxy http://localhost:7897
  ./convert.sh --seq-len 256
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform)  PLATFORM="$2"; shift 2 ;;
        -o|--outdir)    OUTDIR="$2"; shift 2 ;;
        -n|--name)      NAME="$2"; shift 2 ;;
        -L|--seq-len)   SEQ_LEN="$2"; shift 2 ;;
        --quant)        QUANT="--quant"; shift ;;
        --dataset)      DATASET="$2"; shift 2 ;;
        --modelscope)   MODELSCOPE="--modelscope"; shift ;;
        --proxy)        PROXY="$2"; shift 2 ;;
        --hf-endpoint)  HF_ENDPOINT_ARG="$2"; shift 2 ;;
        --py)           PY="$2"; shift 2 ;;
        --skip-onnx)    SKIP_ONNX=1; shift ;;
        --skip-rknn)    SKIP_RKNN=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        -*)             echo "未知选项: $1" >&2; usage; exit 2 ;;
        *)              MODEL="$1"; shift ;;
    esac
done

ONNX="$OUTDIR/$NAME.onnx"
RKNN="$OUTDIR/$NAME.rknn"

# ---- 前置检查 ----
if [[ ! -x "$PY" ]]; then
    echo "✗ 找不到 python: $PY (用 --py 指定或建仓库根 .venv)" >&2; exit 1
fi
"$PY" -c "from rknn.api import RKNN" 2>/dev/null \
    || { echo "✗ venv 里没装 full toolkit (rknn.api 导入失败)" >&2; exit 1; }

export PYTHONPATH="$REPO_ROOT"
if [[ -n "$PROXY" ]]; then export http_proxy="$PROXY" https_proxy="$PROXY"; fi
if [[ -n "$HF_ENDPOINT_ARG" ]]; then export HF_ENDPOINT="$HF_ENDPOINT_ARG"; fi
mkdir -p "$OUTDIR"

echo "════════════════════════════════════════════════════════"
echo " [bge-reranker-v2-m3] 模型: $MODEL   平台: $PLATFORM   seq_len: $SEQ_LEN"
echo " venv: $PY"
echo " 量化: ${QUANT:-fp16(不量化)}"
echo " 输出: $OUTDIR/$NAME.{onnx,rknn} + $OUTDIR/tokenizer/"
echo "════════════════════════════════════════════════════════"

cd "$PY_DIR"

if [[ "$SKIP_ONNX" -eq 0 ]]; then
    echo ""; echo "[1/2] export_onnx.py → ONNX + tokenizer ..."
    t0=$SECONDS
    "$PY" export_onnx.py --model_path "$MODEL" --export_onnx_path "$ONNX" \
        --seq_len "$SEQ_LEN" $MODELSCOPE
    echo "  ✓ export_onnx 完成 ($((SECONDS - t0))s)"
else
    echo "[1/2] 跳过 export_onnx (--skip-onnx)"
    [[ -f "$ONNX" ]] || { echo "✗ 缺少 $ONNX" >&2; exit 1; }
fi

if [[ "$SKIP_RKNN" -eq 0 ]]; then
    echo ""; echo "[2/2] export_rknn.py --platform $PLATFORM → .rknn ..."
    t0=$SECONDS
    DS_ARG=""; [[ -n "$DATASET" ]] && DS_ARG="--dataset $DATASET"
    "$PY" export_rknn.py --onnx_path "$ONNX" --rknn_path "$RKNN" \
        --platform "$PLATFORM" --seq_len "$SEQ_LEN" $QUANT $DS_ARG
    echo "  ✓ export_rknn 完成 ($((SECONDS - t0))s)"
else
    echo "[2/2] 跳过 export_rknn (--skip-rknn)"
fi

echo ""; echo "════════════════════════════════════════════════════════"
echo " 产物 ($OUTDIR):"
for f in "$NAME.onnx" "$NAME.rknn" "$NAME.weight" "$NAME.embed.bin"; do
    [[ -f "$OUTDIR/$f" ]] && printf "   ✓ %-28s %s\n" "$f" "$(du -h "$OUTDIR/$f" | cut -f1)" \
                          || printf "   · %-28s (未生成)\n" "$f"
done
[[ -d "$OUTDIR/tokenizer" ]] && printf "   ✓ %-28s %s\n" "tokenizer/" "$(du -sh "$OUTDIR/tokenizer" | cut -f1)"
echo "════════════════════════════════════════════════════════"
echo " 部署到板子需要: .rknn + .weight + .embed.bin + tokenizer/  (按 seq_len=$SEQ_LEN padding)"
echo " host 端:用 .embed.bin 查表得 inputs_embeds,并预算 position_ids 后喂入(见 README)"
