"""Diagnostic: export bge encoder that takes inputs_embeds (NO word-embedding Gather).
If this LOADS on the board, the 250002x1024 embedding Gather is what the firmware rejects
at MODEL_SETUP (same reason LLM path does embed lookup on host). Tiny weight -> fast iterate.
"""
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from rknn.api import RKNN


class W(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, inputs_embeds, attention_mask, position_ids):
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                          position_ids=position_ids, return_dict=True).logits.view(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--name", default="bge-embeds")
    args = ap.parse_args()

    mp = "BAAI/bge-reranker-v2-m3"
    cfg = AutoConfig.from_pretrained(mp)
    model = AutoModelForSequenceClassification.from_pretrained(mp, config=cfg, torch_dtype=torch.float32)
    model.roberta.encoder.layer = model.roberta.encoder.layer[:args.layers]
    model.config.num_hidden_layers = args.layers
    model.eval()
    H = cfg.hidden_size
    L = args.seq_len

    embeds = torch.zeros(1, L, H, dtype=torch.float32)
    mask = torch.zeros(1, L, dtype=torch.int64); mask[0, :5] = 1
    pos = torch.arange(L, dtype=torch.int64).unsqueeze(0) + 1

    onnx_path = f"../model/{args.name}.onnx"
    torch.onnx.export(W(model), (embeds, mask, pos), onnx_path,
                      input_names=["inputs_embeds", "attention_mask", "position_ids"],
                      output_names=["score"], opset_version=17, do_constant_folding=True)
    print(f"--> onnx exported ({args.layers} layers, seq_len={L}, inputs_embeds)", flush=True)

    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk1828")
    assert rknn.load_onnx(model=onnx_path,
                          inputs=["inputs_embeds", "attention_mask", "position_ids"],
                          input_size_list=[[1, L, H], [1, L], [1, L]]) == 0
    assert rknn.build(do_quantization=False) == 0
    assert rknn.export_rknn(f"../model/{args.name}.rknn") == 0
    rknn.release()
    print(f"--> rknn exported -> ../model/{args.name}.rknn", flush=True)


if __name__ == "__main__":
    main()
