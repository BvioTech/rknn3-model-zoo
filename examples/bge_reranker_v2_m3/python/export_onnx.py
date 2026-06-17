import os
# 镜像走环境变量 HF_ENDPOINT(不带结尾斜杠);默认 https://huggingface.co。
# 不硬编码 hf-mirror(对部分文件 308 跳回 hf.co 会失败)。
import sys
import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


# bge-reranker-v2-m3 是 XLM-RoBERTa encoder + 单 logit 分类头:
#   输入  input_ids[1,L], attention_mask[1,L]  (int64)
#   输出  logits[1,1]                           (相关性分数,sigmoid 后为 0~1)
# 与 Qwen3-Reranker(decoder-only,走 load_llm 流式)架构完全不同,这里走标准 ONNX 路线。

class RerankerWrapper(torch.nn.Module):
    """只暴露 (inputs_embeds, attention_mask, position_ids) -> score[1]。

    关键(RK1828 固件约束):**词向量查表(word_embeddings Gather, 250002x1024)放到 host 端做**,
    模型图以 inputs_embeds 为输入,图里**没有大表 Gather**。原因:RK1828 NPU 固件无法在片上做
    这么大的 embedding Gather,带它的模型会在 model_init 阶段 MODEL_SETUP fail(实测;小表 Gather
    如 position/token_type 514x1024 没问题)。这与 LLM 路线用 .embed.bin + host 回调查表同理。

    position_ids 也作为显式输入喂入(去掉内部从 attention_mask 推 position 的 CumSum;CumSum 落 CPU
    段同样会让 RK1828 MODEL_SETUP fail)。两者都由板端 host 预算后喂入(见 README)。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                         position_ids=position_ids, return_dict=True)
        # logits: [batch, 1] -> [batch]
        return out.logits.view(-1)


def make_position_ids(input_ids, padding_idx):
    """复刻 transformers create_position_ids_from_input_ids:
        mask = (input_ids != padding_idx); pos = cumsum(mask)*mask + padding_idx
    板端推理须用**同一**公式预算 position_ids。"""
    mask = input_ids.ne(padding_idx).to(torch.int64)
    return torch.cumsum(mask, dim=1) * mask + padding_idx


def main():
    parser = argparse.ArgumentParser(
        description="Export BAAI/bge-reranker-v2-m3 (XLM-RoBERTa reranker) to ONNX for RKNN")
    parser.add_argument("--model_path", type=str, default="BAAI/bge-reranker-v2-m3",
                        help="HF 模型 id 或本地路径")
    parser.add_argument("--export_onnx_path", type=str, default="../model/bge-reranker-v2-m3.onnx",
                        help="导出的 onnx 路径")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="固定序列长度(RKNN 需要静态 shape;板端按此长度 padding)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--modelscope", action="store_true", help="从 modelscope.cn 下载")
    parser.add_argument("--save_tokenizer", action="store_true", default=True,
                        help="同时保存 HF tokenizer 到 onnx 同目录的 tokenizer/ 下(板端用)")
    args = parser.parse_args()

    model_path = args.model_path
    if args.modelscope:
        from modelscope import snapshot_download
        model_path = snapshot_download(model_path)

    print(f"--> Loading {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()

    wrapper = RerankerWrapper(model)

    L = args.seq_len
    # 用真实分词构造一个有意义的 dummy 输入(query / document 对),再 pad/truncate 到 L。
    enc = tokenizer(
        "what is panda?",
        "The giant panda is a bear species endemic to China.",
        padding="max_length", truncation=True, max_length=L, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.int64)
    attention_mask = enc["attention_mask"].to(torch.int64)
    padding_idx = config.pad_token_id if config.pad_token_id is not None else 1
    position_ids = make_position_ids(input_ids, padding_idx)

    out_dir = os.path.dirname(os.path.abspath(args.export_onnx_path))
    os.makedirs(out_dir, exist_ok=True)

    # 词向量表(host 查表用):导出为 fp16 raw,板端按 [vocab, hidden] 读取后用 input_ids 索引。
    word_emb = model.roberta.embeddings.word_embeddings.weight.detach()  # [vocab, hidden]
    vocab, hidden = word_emb.shape
    embed_path = os.path.join(out_dir, os.path.basename(args.export_onnx_path).rsplit(".", 1)[0] + ".embed.bin")
    word_emb.to(torch.float16).cpu().numpy().tofile(embed_path)
    print(f"--> Word-embedding table saved to {embed_path}  (vocab={vocab}, hidden={hidden}, fp16)")

    # dummy inputs_embeds = 用 host 查表(与板端一致):word_emb[input_ids]
    inputs_embeds = torch.nn.functional.embedding(input_ids, word_emb).to(torch.float32)

    with torch.no_grad():
        ref = wrapper(inputs_embeds, attention_mask, position_ids)
    print(f"  sanity score (raw logit) = {ref.tolist()}  sigmoid = {torch.sigmoid(ref).tolist()}")
    print(f"  padding_idx={padding_idx} (position_ids 显式输入,板端须用同公式预算)")

    print(f"--> Exporting ONNX (seq_len={L}, hidden={hidden}, opset={args.opset}) -> {args.export_onnx_path}")
    torch.onnx.export(
        wrapper,
        (inputs_embeds, attention_mask, position_ids),
        args.export_onnx_path,
        input_names=["inputs_embeds", "attention_mask", "position_ids"],
        output_names=["score"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,  # 固定 shape;RKNN 静态图
    )
    print("  done")

    if args.save_tokenizer:
        tok_dir = os.path.join(out_dir, "tokenizer")
        tokenizer.save_pretrained(tok_dir)
        print(f"--> Tokenizer saved to {tok_dir}")

    print("\n板端推理需要: .rknn + .weight + {} + tokenizer/  (max_length={} padding)".format(
        os.path.basename(embed_path), L))
    print("  host 端用 .embed.bin 查表得 inputs_embeds,并用 make_position_ids 预算 position_ids 后喂入")


if __name__ == "__main__":
    main()
