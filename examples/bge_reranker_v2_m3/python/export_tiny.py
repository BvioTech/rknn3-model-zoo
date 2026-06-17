"""Diagnostic: export a TRUNCATED bge-reranker (few layers, short seq_len) to ONNX+RKNN
to bisect why the full model fails MODEL_SETUP on RK1828.
  - if a 2-layer model LOADS  -> it's a scale/command-size limit (24 layers too big for 1 core)
  - if a 2-layer model FAILS  -> a per-layer op is unsupported by the firmware
"""
import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from rknn.api import RKNN


class W(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, input_ids, attention_mask, position_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          position_ids=position_ids, return_dict=True).logits.view(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--name", default="bge-tiny")
    args = ap.parse_args()

    mp = "BAAI/bge-reranker-v2-m3"
    cfg = AutoConfig.from_pretrained(mp)
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForSequenceClassification.from_pretrained(mp, config=cfg, torch_dtype=torch.float32)
    # truncate encoder layers
    enc = model.roberta.encoder
    enc.layer = enc.layer[:args.layers]
    model.config.num_hidden_layers = args.layers
    model.eval()
    pad = cfg.pad_token_id if cfg.pad_token_id is not None else 1

    L = args.seq_len
    e = tok("what is panda?", "The giant panda is a bear endemic to China.",
            padding="max_length", truncation=True, max_length=L, return_tensors="pt")
    ids = e["input_ids"].to(torch.int64)
    mask = e["attention_mask"].to(torch.int64)
    m = ids.ne(pad).to(torch.int64)
    pos = torch.cumsum(m, dim=1) * m + pad

    onnx_path = f"../model/{args.name}.onnx"
    torch.onnx.export(W(model), (ids, mask, pos), onnx_path,
                      input_names=["input_ids", "attention_mask", "position_ids"],
                      output_names=["score"], opset_version=17, do_constant_folding=True)
    print(f"--> onnx exported ({args.layers} layers, seq_len={L})", flush=True)

    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk1828")
    assert rknn.load_onnx(model=onnx_path,
                          inputs=["input_ids", "attention_mask", "position_ids"],
                          input_size_list=[[1, L], [1, L], [1, L]]) == 0
    assert rknn.build(do_quantization=False) == 0
    assert rknn.export_rknn(f"../model/{args.name}.rknn") == 0
    rknn.release()
    print(f"--> rknn exported -> ../model/{args.name}.rknn", flush=True)


if __name__ == "__main__":
    main()
