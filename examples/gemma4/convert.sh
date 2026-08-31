#!/usr/bin/env bash
# convert.sh — Gemma-4 LLM 一键转换(PC 端,x86_64,无需目标板)。仅 LLM 模态。
#
#   1) llm/export_llm.py   HF 模型 → .onnx / .config.pkl / .tokenizer.gguf / .embed.bin
#                          (--quant 时先用 rknn.quantization.api.RKQuantizer 做 GRQ 伪量化,需 CUDA)
#   2) llm/export_rknn.py  .onnx → .rknn / .weight  (w4a16 / normal / group32)
#                          + .runtime.json(上下文长度 / 采样参数 / KV 配置,板端启动时按它取值)
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
MAX_CTX=""              # 留空则用 export_rknn.py 的默认值 (16384)
SAMPLING_ARGS=()        # --top-k / --top-p / ... 透传给 export_rknn.py
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
      --max-ctx N     最大上下文长度 (默认 16384;KV cache 按此值静态分配)
      --top-k N       采样 top_k (默认 1 = greedy)
      --top-p F       采样 top_p (默认 0.9)
      --temperature F 采样温度 (默认 1.0)
      --repeat-penalty F / --frequency-penalty F / --presence-penalty F
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
  ./convert.sh -t e4b --max-ctx 16384 --top-k 1        # 显式钉死运行期参数

转换期参数会写进 <outdir>/<name>.runtime.json,部署时随 .rknn 一起拷到板上,
板端启动脚本从该文件读 max_context_len / 采样参数,避免和 cpp 里的硬编码值漂移。
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
        --max-ctx)           MAX_CTX="$2"; shift 2 ;;
        --top-k)             SAMPLING_ARGS+=(--top_k "$2"); shift 2 ;;
        --top-p)             SAMPLING_ARGS+=(--top_p "$2"); shift 2 ;;
        --temperature)       SAMPLING_ARGS+=(--temperature "$2"); shift 2 ;;
        --repeat-penalty)    SAMPLING_ARGS+=(--repeat_penalty "$2"); shift 2 ;;
        --frequency-penalty) SAMPLING_ARGS+=(--frequency_penalty "$2"); shift 2 ;;
        --presence-penalty)  SAMPLING_ARGS+=(--presence_penalty "$2"); shift 2 ;;
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
RUNTIME_JSON="$OUTDIR/$NAME.runtime.json"
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
echo " 输出: $OUTDIR/$NAME.{onnx,config.pkl,tokenizer.gguf,embed.bin,rknn,weight,runtime.json}"
echo "════════════════════════════════════════════════════════"

cd "$LLM_DIR"

if [[ "$SKIP_ONNX" -eq 0 ]]; then
    echo ""; echo "[1/2] export_llm.py → ONNX / config.pkl / tokenizer.gguf / embed.bin ..."
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
    # export_rknn 走 quantized_algorithm='normal'(纯权重量化),不需要校准集;
    # datasets/CMMLU/dataset.txt 是 gitignore 的生成物,不存在时别传空路径给 toolkit。
    DATASET_ARG=()
    if [[ -f "$DATASET" ]]; then DATASET_ARG=(--dataset_path "$DATASET"); else
        echo "  · 未找到 $DATASET,按无校准集(weight-only)量化"
    fi
    "$PY" export_rknn.py --model_type "$MTYPE" --platform "$PLATFORM" \
        --onnx_path "$ONNX" --config "$CONFIG" --rknn_path "$RKNN" ${DATASET_ARG[@]+"${DATASET_ARG[@]}"} \
        --model_path "$MODEL" --runtime_json "$RUNTIME_JSON" \
        ${MAX_CTX:+--max_context_len "$MAX_CTX"} ${SAMPLING_ARGS[@]+"${SAMPLING_ARGS[@]}"}
    echo "  ✓ export_rknn 完成 ($((SECONDS - t0))s)"
else
    echo "[2/2] 跳过 export_rknn (--skip-rknn)"
fi

echo ""; echo "════════════════════════════════════════════════════════"
echo " 产物 ($OUTDIR):"
for ext in onnx config.pkl tokenizer.gguf embed.bin _per_layer_inputs.embed.bin rknn weight runtime.json; do
    case "$ext" in _*) f="$OUTDIR/$NAME$ext" ;; *) f="$OUTDIR/$NAME.$ext" ;; esac
    [[ -f "$f" ]] && printf "   ✓ %-30s %s\n" "$(basename "$f")" "$(du -h "$f" | cut -f1)" \
                  || printf "   · %-30s (未生成)\n" "$(basename "$f")"
done
echo "════════════════════════════════════════════════════════"
echo " 部署到板子需要: .rknn  .weight  .embed.bin  _per_layer_inputs.embed.bin  .runtime.json  + tokenizer"
if [[ -f "$RUNTIME_JSON" ]]; then
    echo ""
    echo " 运行期参数 ($NAME.runtime.json):"
    "$PY" - "$RUNTIME_JSON" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
c, sp = d['context'], d['sampling']
print(f"   max_context_len : {c['max_context_len']}  (板端 <max_context_len> 取等)")
print(f"   sliding_window  : {c['sliding_window']}   prefill_chunk: {c['prefill_chunk_len']}")
print("   sampling        : " + "  ".join(f"{k}={v}" for k, v in sp.items()))
PYEOF
fi
