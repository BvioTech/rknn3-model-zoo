# bge-reranker-v2-m3 (RKNN3 / RK1828)

将 [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) 转成 RKNN,在 RK1828 上做 query-document 重排序。

## 与 Qwen3_Reranker 的区别

| | bge-reranker-v2-m3 | Qwen3-Reranker |
|---|---|---|
| 架构 | XLM-RoBERTa **encoder** + 单 logit 分类头 | Qwen3 **decoder-only** |
| 转换路线 | `rknn.load_onnx` + `build`(标准 ONNX) | `rknn.load_llm`(KV-cache 流式) |
| 输入 | `input_ids[1,L]` + `attention_mask[1,L]`,固定 L | 变长 token 流 |
| 输出 | 一个相关性 logit(sigmoid → 0~1) | 同左,但靠生成 yes/no logit |
| venv | 复用仓库根 `.venv`(torch 2.7 / tf 4.51.3) | 同左 |

因为是 encoder + 静态 shape,这里**不需要** `.venv-vl` / `.venv-gemma`,也不走 LLM 那套量化。

## 用法

```bash
# 仓库根需已有 .venv(full toolkit)
./convert.sh                                  # 默认 fp16,seq_len=512,平台 rk1828
./convert.sh --proxy http://localhost:7897    # 走代理下载 HF 权重
./convert.sh --seq-len 256                    # 更短上下文,更省内存/更快
./convert.sh --skip-onnx                      # 复用已有 onnx 只转 rknn
```

产物(`model/` 下):

- `bge-reranker-v2-m3.onnx` — 中间 ONNX(固定 shape;权重走 external data,同目录另有一堆权重文件)
- `bge-reranker-v2-m3.rknn` — 板端模型结构(~1.2M)
- `bge-reranker-v2-m3.weight` — 板端权重(fp16,~1.1G),**与 .rknn 配套,缺一不可**
- `tokenizer/` — HF tokenizer,板端按 `seq_len` 做 `max_length` padding

> 板端 `load_rknn(rknn_path, weight_path)` 需同时传 `.rknn` 和 `.weight`。

## 关键点 / 坑

- **固定序列长度**:RKNN 需静态 shape。`export_onnx.py --seq_len` 与 `export_rknn.py --seq_len` 必须一致(convert.sh 用 `-L` 统一传)。板端推理时 query+doc 拼接后按同一长度 padding/truncation。
- **平台默认 rk1828**:toolkit 原始默认是 rk1820,本仓库脚本已显式默认 rk1828,用 `-p` 可改。
- **fp16 vs int8**:默认 fp16(`do_quantization=False`),排序分数最接近原模型且无需校准集。如需 int8 减体积,用 `--quant --dataset <list.txt>`(列表每行为 tokenized 后的 npy 路径),但单 logit 排序对量化敏感,务必验证 nDCG/Top-k 再用。
- **下载**:走环境变量 `HF_ENDPOINT`(默认 hf.co),不要用 hf-mirror(部分文件 308 跳回 hf.co 会失败);或加 `--proxy`。
- **CumSum 落 CPU**:XLM-RoBERTa 用 `CumSum`(从 attention_mask 推 position_ids)算子 RKNN 不支持,转换时会提示 `Setting unsupported op: CumSum ... to CPU target`。转换不报错,但板端 runtime 必须支持该 op 在 CPU 执行(本仓库 lite runtime 已支持);如板端报缺 op,可改为在 host 端预算 position_ids 后作为额外输入喂入。
- **PC 模拟器验证**:`cd python && python test_pc.py`。注意 `load_rknn` 加载的模型不能在模拟器推理,故 test_pc.py 走 `load_onnx+build` 重建一次再推理(与转换链路一致)。实测 fp16 与原模型分数逐位吻合(panda 例:logit 1.646 vs 1.654)。

## 板端推理(思路)

bge reranker 推理无 KV-cache,单次前向:

1. host 上用 `tokenizer/` 把 `(query, doc)` 编码成 `input_ids`/`attention_mask`,padding 到 `seq_len`。
2. `rknn.inference(inputs=[input_ids, attention_mask])` → 得到 logit。
3. `score = sigmoid(logit)`,对候选 doc 排序。

批量重排序即对每个候选各跑一次前向取分排序。
