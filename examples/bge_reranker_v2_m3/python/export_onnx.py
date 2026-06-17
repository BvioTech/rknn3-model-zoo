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
    """只暴露 (input_ids, attention_mask) -> score[1],去掉 HF 输出对象,方便 ONNX 导出 / RKNN 加载。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # logits: [batch, 1] -> [batch]
        return out.logits.view(-1)


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

    out_dir = os.path.dirname(os.path.abspath(args.export_onnx_path))
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        ref = wrapper(input_ids, attention_mask)
    print(f"  sanity score (raw logit) = {ref.tolist()}  sigmoid = {torch.sigmoid(ref).tolist()}")

    print(f"--> Exporting ONNX (seq_len={L}, opset={args.opset}) -> {args.export_onnx_path}")
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        args.export_onnx_path,
        input_names=["input_ids", "attention_mask"],
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

    print("\n板端推理需要: 上面的 .rknn  +  tokenizer/  (HF tokenizer, max_length={} padding)".format(L))


if __name__ == "__main__":
    main()
