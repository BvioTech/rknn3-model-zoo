# bge-reranker-v2-m3 (RKNN3 / RK1828)

将 [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) 转成 RKNN,在 RK1828 上做 query-document 重排序。

## 与 Qwen3_Reranker 的区别

| | bge-reranker-v2-m3 | Qwen3-Reranker |
|---|---|---|
| 架构 | XLM-RoBERTa **encoder** + 单 logit 分类头 | Qwen3 **decoder-only** |
| 转换路线 | `rknn.load_onnx` + `build`(标准 ONNX) | `rknn.load_llm`(KV-cache 流式) |
| 模型图输入 | `inputs_embeds[1,L,H]` + `attention_mask[1,L]` + `position_ids[1,L]`,固定 L | 变长 token 流 |
| 输出 | 一个相关性 logit(sigmoid → 0~1) | 同左,但靠生成 yes/no logit |
| venv | 复用仓库根 `.venv`(torch 2.7 / tf 4.51.3) | 同左 |

因为是 encoder + 静态 shape,这里**不需要** `.venv-vl` / `.venv-gemma`,也不走 LLM 那套量化。

### ⚠️ RK1828 固件约束(为什么是 inputs_embeds 而不是 input_ids)

直接喂 `input_ids` 的整图(含 250002×1024 词向量大表 `Gather`)在 RK1828 上 **`MODEL_SETUP` 失败**——固件无法在片上做这么大的 embedding Gather(与 LLM 路线用 `.embed.bin` + host 回调查表同因)。所以本目录的做法:

1. **词向量查表放 host 端**:模型图改以 `inputs_embeds` 为输入,词表导出成 `.embed.bin`(fp16,`[vocab, hidden]`)。板端 host 用 `input_ids` 索引 `.embed.bin` 得 `inputs_embeds` 再喂入。
2. **`position_ids` 显式输入**:去掉内部从 `attention_mask` 推 position 的 `CumSum`(落 CPU 段也会让 `MODEL_SETUP` fail)。host 用 `make_position_ids` 同公式预算后喂入。
3. **多核**:24 层 transformer 单核会 `MODEL_SETUP` fail,`export_rknn.py --core-num` 默认 8(板端 `init_runtime` 的 `core_mask` 须匹配,如 8 核=`0xff`)。

## 用法

```bash
# 仓库根需已有 .venv(full toolkit)
./convert.sh                                  # 默认 fp16,seq_len=512,平台 rk1828
./convert.sh --proxy http://localhost:7897    # 走代理下载 HF 权重
./convert.sh --seq-len 256                    # 更短上下文,更省内存/更快
./convert.sh --skip-onnx                      # 复用已有 onnx 只转 rknn
```

产物(`model/` 下):

- `bge-reranker-v2-m3.onnx` — 中间 ONNX(以 inputs_embeds 为输入;权重走 external data,同目录另有一堆权重文件)
- `bge-reranker-v2-m3.rknn` — 板端模型结构(~1.2M)
- `bge-reranker-v2-m3.weight` — 板端权重(fp16,~600M,不含词表),**与 .rknn 配套,缺一不可**
- `bge-reranker-v2-m3.embed.bin` — 词向量表(fp16,`[vocab, hidden]`,~489M),**host 端查表用,板端必需**
- `tokenizer/` — HF tokenizer,板端按 `seq_len` 做 `max_length` padding

> 板端 `load_rknn(rknn_path, weight_path)` 需同时传 `.rknn` 和 `.weight`;`.embed.bin` 由 host 加载做查表。

## 关键点 / 坑

- **固定序列长度**:RKNN 需静态 shape。`export_onnx.py --seq_len` 与 `export_rknn.py --seq_len` 必须一致(convert.sh 用 `-L` 统一传)。板端推理时 query+doc 拼接后按同一长度 padding/truncation。
- **平台默认 rk1828**:toolkit 原始默认是 rk1820,本仓库脚本已显式默认 rk1828,用 `-p` 可改。
- **fp16 vs int8**:默认 fp16(`do_quantization=False`),排序分数最接近原模型且无需校准集。如需 int8 减体积,用 `--quant --dataset <list.txt>`(列表每行为 tokenized 后的 npy 路径),但单 logit 排序对量化敏感,务必验证 nDCG/Top-k 再用。
- **下载**:走环境变量 `HF_ENDPOINT`(默认 hf.co),不要用 hf-mirror(部分文件 308 跳回 hf.co 会失败);或加 `--proxy`。
- **host 端两件事**(见上面 RK1828 固件约束):① 用 `.embed.bin` 把 `input_ids` 查表成 `inputs_embeds`;② 用 `make_position_ids`(`cumsum(ids!=pad)*(ids!=pad)+pad`)预算 `position_ids`。板端必须用**同一**公式,否则结果错。
- **诊断脚本**:`export_tiny.py` / `export_embeds.py` 是定位 RK1828 `MODEL_SETUP` 失败用的二分脚本(截断层数 / 改 inputs_embeds 快速迭代),记录了得出上述结论的过程,非转换主流程。
- **PC 模拟器验证**:`cd python && python test_pc.py`。注意 `load_rknn` 加载的模型不能在模拟器推理,故 test_pc.py 走 `load_onnx+build` 重建一次再推理(与转换链路一致),并在 host 端用 `.embed.bin` 查表。实测 fp16 与原模型分数逐位吻合(panda 例:logit 1.646 vs 1.654;无关 doc → 0.000)。

## 板端推理(思路)

bge reranker 推理无 KV-cache,单次前向:

1. host 上用 `tokenizer/` 把 `(query, doc)` 编码成 `input_ids`/`attention_mask`,padding 到 `seq_len`。
2. host 用 `.embed.bin` 查表得 `inputs_embeds`,并预算 `position_ids`。
3. `rknn.inference(inputs=[inputs_embeds, attention_mask, position_ids])` → 得到 logit。
4. `score = sigmoid(logit)`,对候选 doc 排序。

批量重排序即对每个候选各跑一次前向取分排序。
