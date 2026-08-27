#!/usr/bin/env bash
# convert.sh — Gemma-4 LLM 一键转换(PC 端,x86_64,无需目标板)。仅 LLM 模态。
#
#   1) llm/export_llm.py   HF 模型 → .onnx / .config.pkl / .tokenizer.gguf / .embed.bin (+ 用 gemma
#                          tokenizer 重新生成 datasets/CMMLU 量化数据集)
#   2) llm/export_rknn.py  .onnx → .rknn / .weight  (w4a16 / normal / group32)
#
# ⚠️ Gemma-4 依赖与主仓库及 VL 都不同,需独立 venv:
#      transformers==5.5.0 + torch==2.9.0 + torchvision==0.24.0 + onnxruntime + onnxscript。
#    用 --py 或环境变量 GEMMA_PY 指定;默认找 <repo>/.venv-gemma/bin/python。
#    建独立 venv:
#      python3.12 -m venv .venv-gemma && . .venv-gemma/bin/activate
#      pip install torch==2.9.0 torchvision==0.24.0 transformers==5.5.0 onnxruntime onnxscript
#      pip install --no-deps <full-toolkit.whl>   # --no-deps 防止把 torch 拽回 2.7
#
# 说明:vision / audio 两个模态未纳入本脚本(本次只要 LLM)。
# 详见 examples/gemma4/README.md。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLM_DIR="$SCRIPT_DIR/python/llm"
PY="${GEMMA_PY:-$REPO_ROOT/.venv-gemma/bin/python}"

MTYPE="e4b"             # e2b | e4b
MODEL_OVERRIDE=""       # 本地权重目录或自定义 repo id
PLATFORM="rk1828"
OUTDIR="$SCRIPT_DIR/model/llm"
QUANT=""
MODELSCOPE=""
SKIP_ONNX=0
SKIP_RKNN=0
PROXY="${https_proxy:-}"
HF_ENDPOINT_ARG=""

usage() {
    cat <<'EOF'
用法: convert.sh [选项]

选项:
  -t, --model-type T  e2b | e4b (默认 e4b)
  -m, --model PATH    本地权重目录或自定义 repo id (默认按 -t 取 google/gemma-4-Exx-it)
  -p, --platform P    目标平台 (默认 rk1828;原脚本写死 rk1820)
      --py PATH       Gemma 专用 venv 的 python (默认 <repo>/.venv-gemma/bin/python 或 $GEMMA_PY)
  -o, --outdir DIR    产物输出目录 (默认 ./model/llm)
      --quant         AWQ+GRQ 预量化 (需 CUDA)
      --modelscope    从 modelscope.cn 下载
      --proxy URL     HF 下载代理 (如 http://localhost:7897)
      --hf-endpoint URL  HF 镜像 (不带结尾斜杠)
      --skip-onnx     复用已有 onnx,只转 RKNN
      --skip-rknn     只导出 ONNX
  -h, --help

示例:
  ./convert.sh --model-type e4b
  ./convert.sh -t e4b -m /root/autodl-tmp/gemma-4-E4B-it --quant   # 用已下好的本地权重
  ./convert.sh --py ../../.venv-gemma/bin/python -t e4b --proxy http://localhost:7897
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--model-type) MTYPE="$2"; shift 2 ;;
        -m|--model)      MODEL_OVERRIDE="$2"; shift 2 ;;
        -p|--platform)   PLATFORM="$2"; shift 2 ;;
        --py)            PY="$2"; shift 2 ;;
        -o|--outdir)     OUTDIR="$2"; shift 2 ;;
        --quant)         QUANT="--quant"; shift ;;
        --modelscope)    MODELSCOPE="--modelscope"; shift ;;
        --proxy)         PROXY="$2"; shift 2 ;;
        --hf-endpoint)   HF_ENDPOINT_ARG="$2"; shift 2 ;;
        --skip-onnx)     SKIP_ONNX=1; shift ;;
        --skip-rknn)     SKIP_RKNN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        -*)              echo "未知选项: $1" >&2; usage; exit 2 ;;
        *)               echo "未知参数: $1" >&2; usage; exit 2 ;;
    esac
done

case "$MTYPE" in
    e2b) MODEL="google/gemma-4-E2B-it"; NAME="gemma-4-e2b-it" ;;
    e4b) MODEL="google/gemma-4-E4B-it"; NAME="gemma-4-e4b-it" ;;
    *)   echo "✗ --model-type 只能是 e2b 或 e4b" >&2; exit 2 ;;
esac
[[ -n "$MODEL_OVERRIDE" ]] && MODEL="$MODEL_OVERRIDE"

ONNX="$OUTDIR/$NAME.onnx"
CONFIG="$OUTDIR/$NAME.config.pkl"
RKNN="$OUTDIR/$NAME.rknn"
DATASET="$REPO_ROOT/datasets/CMMLU/dataset.txt"

# ---- 前置检查 ----
if [[ ! -x "$PY" ]]; then
    echo "✗ 找不到 Gemma venv 的 python: $PY" >&2
    echo "  Gemma 需独立环境(transformers==5.5.0 + torch==2.9.0)。用 --py 指定,或建 .venv-gemma(见脚本头注释)。" >&2
    exit 1
fi
"$PY" -c "from rknn.api import RKNN" 2>/dev/null \
    || { echo "✗ Gemma venv 里没装 full toolkit (rknn.api 导入失败)" >&2; exit 1; }
TFVER="$("$PY" -c "import transformers;print(transformers.__version__)" 2>/dev/null || echo '?')"
[[ "$TFVER" == 5.5.* ]] || echo "⚠️  transformers=$TFVER,Gemma-4 推荐 5.5.0,版本不符可能导致转换失败"

export PYTHONPATH="$REPO_ROOT"
if [[ -n "$PROXY" ]]; then export http_proxy="$PROXY" https_proxy="$PROXY"; fi
if [[ -n "$HF_ENDPOINT_ARG" ]]; then export HF_ENDPOINT="$HF_ENDPOINT_ARG"; fi
mkdir -p "$OUTDIR"

echo "════════════════════════════════════════════════════════"
echo " [Gemma-4 $MTYPE] 模型: $MODEL   平台: $PLATFORM"
echo " venv: $PY (transformers $TFVER)"
echo " 输出: $OUTDIR/$NAME.{onnx,config.pkl,tokenizer.gguf,embed.bin,rknn,weight}"
echo "════════════════════════════════════════════════════════"

cd "$LLM_DIR"

if [[ "$SKIP_ONNX" -eq 0 ]]; then
    echo ""; echo "[1/2] export_llm.py → ONNX / config / tokenizer / embed / dataset(gemma tokenizer) ..."
    t0=$SECONDS
    "$PY" export_llm.py --model_path "$MODEL" --export_llm_path "$ONNX" $QUANT $MODELSCOPE
    echo "  ✓ export_llm 完成 ($((SECONDS - t0))s)"
else
    echo "[1/2] 跳过 export_llm (--skip-onnx)"
    [[ -f "$ONNX" ]] || { echo "✗ 缺少 $ONNX" >&2; exit 1; }
fi

if [[ "$SKIP_RKNN" -eq 0 ]]; then
    echo ""; echo "[2/2] export_rknn.py --model_type $MTYPE --platform $PLATFORM → .rknn / .weight ..."
    echo "      (export_rknn 内部会 chdir 到 model/llm,故传绝对路径规避相对路径失效)"
    t0=$SECONDS
    "$PY" export_rknn.py --model_type "$MTYPE" --platform "$PLATFORM" \
        --onnx_path "$ONNX" --config "$CONFIG" --rknn_path "$RKNN" --dataset_path "$DATASET"
    echo "  ✓ export_rknn 完成 ($((SECONDS - t0))s)"
else
    echo "[2/2] 跳过 export_rknn (--skip-rknn)"
fi

echo ""; echo "════════════════════════════════════════════════════════"
echo " 产物 ($OUTDIR):"
for ext in onnx config.pkl tokenizer.gguf embed.bin rknn weight; do
    f="$OUTDIR/$NAME.$ext"
    [[ -f "$f" ]] && printf "   ✓ %-24s %s\n" "$NAME.$ext" "$(du -h "$f" | cut -f1)" \
                  || printf "   · %-24s (未生成)\n" "$NAME.$ext"
done
echo "════════════════════════════════════════════════════════"
echo " 部署到板子需要: .rknn  .weight  .embed.bin  + tokenizer"
