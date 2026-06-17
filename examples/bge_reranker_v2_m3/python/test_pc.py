"""PC 模拟器验证 bge-reranker-v2-m3 RKNN 转换是否正确。
对照 HF 原模型的相关性分数。需在仓库根 .venv 下运行。

模型图以 inputs_embeds 为输入(词向量查表在 host 端做,见 export_onnx.py):
  host: ids -> 用 .embed.bin 查表得 inputs_embeds[1,L,H];position_ids 用同公式预算。
注意:load_rknn 加载的模型不支持模拟器推理(toolkit 限制),
故这里走 load_onnx + build(do_quantization=False)再 init_runtime,与 export_rknn.py 链路一致。
"""
import argparse
import numpy as np
from rknn.api import RKNN
from transformers import AutoTokenizer


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="../model/bge-reranker-v2-m3.onnx")
    p.add_argument("--embed", default="../model/bge-reranker-v2-m3.embed.bin")
    p.add_argument("--tokenizer", default="../model/tokenizer")
    p.add_argument("--platform", default="rk1828")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--query", default="what is panda?")
    p.add_argument("--docs", nargs="+", default=[
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
    ])
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 1
    # host 端词向量表(与板端一致):fp16 raw,[vocab, hidden]
    embed = np.fromfile(args.embed, dtype=np.float16).reshape(-1, args.hidden)

    rknn = RKNN(verbose=False)
    rknn.config(target_platform=args.platform)
    assert rknn.load_onnx(model=args.onnx, inputs=["inputs_embeds", "attention_mask", "position_ids"],
                          input_size_list=[[1, args.seq_len, args.hidden],
                                           [1, args.seq_len], [1, args.seq_len]]) == 0, "load_onnx failed"
    assert rknn.build(do_quantization=False) == 0, "build failed"
    assert rknn.init_runtime() == 0, "init_runtime (simulator) failed"

    for doc in args.docs:
        enc = tok(args.query, doc, padding="max_length", truncation=True,
                  max_length=args.seq_len, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        # host 查表 -> inputs_embeds[1,L,H]
        embeds = embed[ids[0]].astype(np.float32)[None, :, :]
        # position_ids: cumsum(ids!=pad)*(ids!=pad)+pad —— 与 export_onnx.make_position_ids 一致
        m = (ids != pad).astype(np.int64)
        pos = np.cumsum(m, axis=1) * m + pad
        out = rknn.inference(inputs=[embeds, mask, pos])
        logit = float(np.array(out[0]).reshape(-1)[0])
        print(f"  score={sigmoid(logit):.4f} (logit={logit:+.4f})  doc: {doc}")

    rknn.release()


if __name__ == "__main__":
    main()
