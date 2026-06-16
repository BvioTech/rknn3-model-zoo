#!/usr/bin/env bash
# convert.sh — Qwen3-Embedding 一键转换(PC 端,x86_64,无需目标板)
#
#   1) export_llm.py   HF 模型 → .onnx / .config.pkl(task_type=1) / .tokenizer.gguf / .embed.bin
#   2) export_rknn.py  .onnx → .rknn / .weight  (w4a16 / normal / group32,max_ctx_len=2048)
#
# 与纯 LLM 的区别:输出是 embedding 向量(不是生成文本);量化算法用 normal;
# export_rknn 需要量化数据集 datasets/CMMLU/dataset.txt(缺失时本脚本自动生成)。
# 复用仓库根 .venv(full toolkit,transformers 4.51.3 即可)。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY_DIR="$SCRIPT_DIR/python"
PY="$REPO_ROOT/.venv/bin/python"

MODEL="Qwen/Qwen3-Embedding-0.6B"
PLATFORM="rk1828"
OUTDIR="$SCRIPT_DIR/model/llm"
NAME=""
MODELSCOPE=""
SKIP_ONNX=0
SKIP_RKNN=0
PROXY="${https_proxy:-}"
HF_ENDPOINT_ARG=""

usage() {
    cat <<'EOF'
用法: convert.sh [选项] [<model>]

  <model>            HF 模型 id 或本地路径 (默认 Qwen/Qwen3-Embedding-0.6B)

选项:
  -p, --platform P   目标平台 (默认 rk1828)
  -o, --outdir DIR   产物输出目录 (默认 ./model/llm)
  -n, --name NAME    产物基名 (默认取 model 的 basename)
      --modelscope   从 modelscope.cn 下载
      --proxy URL    HF 下载代理 (如 http://localhost:7897)
      --hf-endpoint URL  HF 镜像 (不带结尾斜杠)
      --skip-onnx    复用已有 onnx,只转 RKNN
      --skip-rknn    只导出 ONNX
  -h, --help

示例:
  ./convert.sh Qwen/Qwen3-Embedding-0.6B
  ./convert.sh --proxy http://localhost:7897 Qwen/Qwen3-Embedding-4B
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform)  PLATFORM="$2"; shift 2 ;;
        -o|--outdir)    OUTDIR="$2"; shift 2 ;;
        -n|--name)      NAME="$2"; shift 2 ;;
        --modelscope)   MODELSCOPE="--modelscope"; shift ;;
        --proxy)        PROXY="$2"; shift 2 ;;
        --hf-endpoint)  HF_ENDPOINT_ARG="$2"; shift 2 ;;
        --skip-onnx)    SKIP_ONNX=1; shift ;;
        --skip-rknn)    SKIP_RKNN=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        -*)             echo "未知选项: $1" >&2; usage; exit 2 ;;
        *)              MODEL="$1"; shift ;;
    esac
done

[[ -z "$NAME" ]] && NAME="$(basename "$MODEL")"
ONNX="$OUTDIR/$NAME.onnx"
CONFIG="$OUTDIR/$NAME.config.pkl"
RKNN="$OUTDIR/$NAME.rknn"
DATASET="$REPO_ROOT/datasets/CMMLU/dataset.txt"

[[ -x "$PY" ]] || { echo "✗ 找不到 venv: $PY (见 docs/rk1828/01 建 .venv 并装 full toolkit)" >&2; exit 1; }
"$PY" -c "from rknn.api import RKNN" 2>/dev/null \
    || { echo "✗ venv 里没装 full toolkit (rknn.api 导入失败)" >&2; exit 1; }

export PYTHONPATH="$REPO_ROOT"
if [[ -n "$PROXY" ]]; then export http_proxy="$PROXY" https_proxy="$PROXY"; fi
if [[ -n "$HF_ENDPOINT_ARG" ]]; then export HF_ENDPOINT="$HF_ENDPOINT_ARG"; fi
mkdir -p "$OUTDIR"

echo "════════════════════════════════════════════════════════"
echo " [Embedding] 模型: $MODEL   平台: $PLATFORM"
echo " 输出: $OUTDIR/$NAME.{onnx,config.pkl,tokenizer.gguf,embed.bin,rknn,weight}"
echo "════════════════════════════════════════════════════════"

cd "$PY_DIR"

if [[ "$SKIP_ONNX" -eq 0 ]]; then
    echo ""; echo "[1/2] export_llm.py → ONNX / config(task_type=1) / tokenizer / embed ..."
    t0=$SECONDS
    "$PY" export_llm.py --model_path "$MODEL" --export_llm_path "$ONNX" $MODELSCOPE
    echo "  ✓ export_llm 完成 ($((SECONDS - t0))s)"
else
    echo "[1/2] 跳过 export_llm (--skip-onnx)"
    [[ -f "$ONNX" ]] || { echo "✗ 缺少 $ONNX" >&2; exit 1; }
fi

# 量化数据集:export_rknn 走 do_quantization=True,必须有 dataset.txt;缺失则用本模型生成
if [[ "$SKIP_RKNN" -eq 0 && ! -f "$DATASET" ]]; then
    echo ""; echo "[*] 量化数据集缺失,用 $MODEL 生成 $DATASET ..."
    "$PY" - "$MODEL" <<'PYGEN'
import sys, os
sys.path.insert(0, os.environ["PYTHONPATH"])
from transformers import AutoModelForCausalLM, AutoConfig
from py_utils.tools import gen_quantize_dataset
root = os.environ["PYTHONPATH"]
model_path = sys.argv[1]
cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(model_path, config=cfg, trust_remote_code=True)
gen_quantize_dataset(model_path, m.get_input_embeddings(),
                     f"{root}/datasets/CMMLU/dataset.json",
                     f"{root}/datasets/CMMLU/dataset.txt",
                     f"{root}/datasets/CMMLU/dataset_np")
print("dataset generated")
PYGEN
fi

if [[ "$SKIP_RKNN" -eq 0 ]]; then
    echo ""; echo "[2/2] export_rknn.py --platform $PLATFORM → .rknn / .weight (normal w4a16 group32) ..."
    t0=$SECONDS
    "$PY" export_rknn.py --platform "$PLATFORM" \
        --onnx_path "$ONNX" --config "$CONFIG" --rknn_path "$RKNN" --dataset_path "$DATASET"
    echo "  ✓ export_rknn 完成 ($((SECONDS - t0))s)"
else
    echo "[2/2] 跳过 export_rknn (--skip-rknn)"
fi

echo ""; echo "════════════════════════════════════════════════════════"
echo " 产物 ($OUTDIR):"
for ext in onnx config.pkl tokenizer.gguf embed.bin rknn weight; do
    f="$OUTDIR/$NAME.$ext"
    [[ -f "$f" ]] && printf "   ✓ %-26s %s\n" "$NAME.$ext" "$(du -h "$f" | cut -f1)" \
                  || printf "   · %-26s (未生成)\n" "$NAME.$ext"
done
echo "════════════════════════════════════════════════════════"
