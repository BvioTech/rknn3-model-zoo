#!/usr/bin/env bash
# convert.sh — Qwen3-VL 一键转换(PC 端,x86_64,无需目标板)
#
# VL 是「vision 编码器 + LLM」两套模型,共四步:
#   V1) vision/export_vision.py   HF → <name>-vision.onnx (+ vision_config.json)
#   V2) vision/export_rknn.py     → <name>-vision.rknn
#   L1) llm/export_llm.py         HF → <name>-llm.onnx / config / tokenizer / embed
#   L2) llm/export_rknn.py        → <name>-llm.rknn
#
# ⚠️ VL 依赖与主仓库冲突,需独立 venv:transformers==4.57.1 + torch>=2.9 + onnxruntime>=1.23.2。
#    用 --py 或环境变量 VL_PY 指定该 venv 的 python;默认找 <repo>/.venv-vl/bin/python。
#    建独立 venv 示例:
#      python3.12 -m venv .venv-vl && . .venv-vl/bin/activate
#      pip install 'torch>=2.9' torchvision 'transformers==4.57.1' 'onnxruntime>=1.23.2' onnxscript
#      pip install --no-deps <full-toolkit.whl>   # --no-deps 防止把 torch 拽回 2.7
#    注意两个易漏依赖:
#      torchvision —— llm 量化数据集走 AutoProcessor，缺它报 "AutoVideoProcessor requires Torchvision"。
#      onnxscript  —— torch>=2.9 的 dynamo ONNX 导出需要，缺它报 "No module named 'onnxscript'"。
#
# ⚠️ 开箱即用仅覆盖 2B:llm/export_rknn.py 的 dynamic_input 按 onnx 文件名含 "2b"/"4b" 选择,
#    且 mrope_section 硬编码为 2B 的 [24,20,20]。转 4B 需按 config.json 改 mrope_section。
# 详见 examples/Qwen3_VL/README.md。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLM_DIR="$SCRIPT_DIR/python/llm"
VIS_DIR="$SCRIPT_DIR/python/vision"

MODEL="Qwen/Qwen3-VL-2B-Instruct"
PLATFORM="rk1828"
PY="${VL_PY:-$REPO_ROOT/.venv-vl/bin/python}"
OUTDIR="$SCRIPT_DIR/model"
NAME=""
IMG_H=384
IMG_W=384
PRUNE=0                  # 默认完整模型(RK1828 大内存,README 2.3);--prune 启用裁剪
QUANT=""
MODELSCOPE=""
ONLY=""                  # vision | llm | "" (两者)
PROXY="${https_proxy:-}"
HF_ENDPOINT_ARG=""

usage() {
    cat <<'EOF'
用法: convert.sh [选项] [<model>]

  <model>            HF 模型 id 或本地路径 (默认 Qwen/Qwen3-VL-2B-Instruct)

选项:
  -p, --platform P   目标平台 (默认 rk1828)
      --py PATH      VL 专用 venv 的 python (默认 <repo>/.venv-vl/bin/python 或 $VL_PY)
  -o, --outdir DIR   产物输出目录 (默认 ./model;onnx 落 ./model/onnx)
  -n, --name NAME    产物基名 (默认取 model 的 basename;须含 2b/4b)
      --img-h N      vision 输入高 (32 的倍数,默认 384)
      --img-w N      vision 输入宽 (32 的倍数,默认 384)
      --prune        启用裁剪模式 (默认完整模型,适合 RK1828)
      --quant        LLM 走 AWQ+GRQ 预量化 (需 CUDA)
      --only STAGE   只跑某阶段: vision | llm
      --modelscope   从 modelscope.cn 下载
      --proxy URL    HF 下载代理 (如 http://localhost:7897)
      --hf-endpoint URL  HF 镜像 (不带结尾斜杠)
  -h, --help

示例:
  ./convert.sh --py ../../.venv-vl/bin/python Qwen/Qwen3-VL-2B-Instruct
  ./convert.sh --only vision --img-h 448 --img-w 448 Qwen/Qwen3-VL-2B-Instruct
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform)  PLATFORM="$2"; shift 2 ;;
        --py)           PY="$2"; shift 2 ;;
        -o|--outdir)    OUTDIR="$2"; shift 2 ;;
        -n|--name)      NAME="$2"; shift 2 ;;
        --img-h)        IMG_H="$2"; shift 2 ;;
        --img-w)        IMG_W="$2"; shift 2 ;;
        --prune)        PRUNE=1; shift ;;
        --quant)        QUANT="--quant"; shift ;;
        --only)         ONLY="$2"; shift 2 ;;
        --modelscope)   MODELSCOPE="--modelscope"; shift ;;
        --proxy)        PROXY="$2"; shift 2 ;;
        --hf-endpoint)  HF_ENDPOINT_ARG="$2"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        -*)             echo "未知选项: $1" >&2; usage; exit 2 ;;
        *)              MODEL="$1"; shift ;;
    esac
done

[[ -z "$NAME" ]] && NAME="$(basename "$MODEL")"
ONNX_DIR="$OUTDIR/onnx"
VIS_ONNX="$ONNX_DIR/$NAME-vision.onnx"
VIS_RKNN="$OUTDIR/$NAME-vision.rknn"
LLM_ONNX="$ONNX_DIR/$NAME-llm.onnx"
LLM_CONFIG="$ONNX_DIR/$NAME-llm.config.pkl"
LLM_RKNN="$OUTDIR/$NAME-llm.rknn"

# ---- 前置检查 ----
if [[ ! -x "$PY" ]]; then
    echo "✗ 找不到 VL venv 的 python: $PY" >&2
    echo "  VL 需独立环境(transformers==4.57.1 + torch>=2.9)。用 --py 指定,或建 .venv-vl(见脚本头注释)。" >&2
    exit 1
fi
"$PY" -c "from rknn.api import RKNN" 2>/dev/null \
    || { echo "✗ VL venv 里没装 full toolkit (rknn.api 导入失败)" >&2; exit 1; }
TFVER="$("$PY" -c "import transformers;print(transformers.__version__)" 2>/dev/null || echo '?')"
[[ "$TFVER" == 4.57.* ]] || echo "⚠️  transformers=$TFVER,VL 推荐 4.57.1,版本不符可能导致转换失败"
case "${NAME,,}" in
    *2b*|*4b*) : ;;
    *) echo "⚠️  名称 '$NAME' 不含 2b/4b,llm/export_rknn 的 dynamic_input 选择会失败;请用 -n 指定含 2b/4b 的名称" >&2 ;;
esac
case "${NAME,,}" in *4b*) echo "⚠️  4B:mrope_section 在 llm/export_rknn.py 仍是 2B 的 [24,20,20],请按 config.json 修改后再转";; esac

export PYTHONPATH="$REPO_ROOT"
if [[ -n "$PROXY" ]]; then export http_proxy="$PROXY" https_proxy="$PROXY"; fi
if [[ -n "$HF_ENDPOINT_ARG" ]]; then export HF_ENDPOINT="$HF_ENDPOINT_ARG"; fi
mkdir -p "$ONNX_DIR" "$OUTDIR"

echo "════════════════════════════════════════════════════════"
echo " [VL] 模型: $MODEL   平台: $PLATFORM   分辨率: ${IMG_W}x${IMG_H}"
echo " venv: $PY (transformers $TFVER)"
echo " 模式: $([[ $PRUNE -eq 1 ]] && echo 裁剪 || echo 完整模型)   阶段: ${ONLY:-vision+llm}"
echo "════════════════════════════════════════════════════════"

run_vision() {
    echo ""; echo "[V1/V2] vision: export_vision.py → onnx (+ vision_config.json) ..."
    cd "$VIS_DIR"
    "$PY" export_vision.py --model_path "$MODEL" --export_vision_path "$VIS_ONNX" \
        --img_h "$IMG_H" --img_w "$IMG_W" $MODELSCOPE
    echo "        export_rknn.py → $NAME-vision.rknn ..."
    local prune_flag=""
    [[ $PRUNE -eq 0 ]] && prune_flag="--no_prune_mode"
    "$PY" export_rknn.py --platform "$PLATFORM" \
        --onnx_path "$VIS_ONNX" --rknn_path "$VIS_RKNN" $prune_flag
    echo "  ✓ vision 完成"
}

run_llm() {
    echo ""; echo "[L1/L2] llm: export_llm.py → onnx / config / tokenizer / embed ..."
    cd "$LLM_DIR"
    "$PY" export_llm.py --model_path "$MODEL" --export_llm_path "$LLM_ONNX" $QUANT $MODELSCOPE
    echo "        export_rknn.py → $NAME-llm.rknn ..."
    "$PY" export_rknn.py --platform "$PLATFORM" \
        --onnx_path "$LLM_ONNX" --config "$LLM_CONFIG" --rknn_path "$LLM_RKNN"
    echo "  ✓ llm 完成"
}

case "$ONLY" in
    vision) run_vision ;;
    llm)    run_llm ;;
    "")     run_vision; run_llm ;;
    *)      echo "✗ --only 只能是 vision 或 llm" >&2; exit 2 ;;
esac

echo ""; echo "════════════════════════════════════════════════════════"
echo " 产物 ($OUTDIR):"
for f in "$VIS_RKNN" "${VIS_RKNN%.rknn}.weight" \
         "$LLM_RKNN" "${LLM_RKNN%.rknn}.weight" \
         "$LLM_ONNX" "$LLM_CONFIG" "${LLM_ONNX%.onnx}.tokenizer.gguf" "${LLM_ONNX%.onnx}.embed.bin"; do
    [[ -f "$f" ]] && printf "   ✓ %-34s %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
done
echo "════════════════════════════════════════════════════════"
echo " 部署到板子需要: vision.rknn(+weight) llm.rknn(+weight) embed.bin tokenizer"
