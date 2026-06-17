import argparse
import os

from rknn.api import RKNN


# bge-reranker-v2-m3:XLM-RoBERTa encoder,走标准 load_onnx 路线(非 load_llm)。
# 输入 inputs_embeds[1,L,H] / attention_mask[1,L] / position_ids[1,L];输出 score[1]。
# 词向量查表(word_embeddings, 250002xH)放 host 端(导出为 .embed.bin),图里以 inputs_embeds
# 为输入——否则 RK1828 固件无法在片上做大表 Gather,会 MODEL_SETUP fail(见 export_onnx.py)。
# position_ids 也显式输入(去掉内部 CumSum)。--seq-len 须与 export_onnx 一致;-H 取模型 hidden。
# 默认 fp16(do_quantization=False):排序精度最接近原模型,板端无需校准数据集。

def main():
    parser = argparse.ArgumentParser(description="Export bge-reranker-v2-m3 ONNX to RKNN")
    parser.add_argument("--onnx_path", type=str, default="../model/bge-reranker-v2-m3.onnx")
    parser.add_argument("--rknn_path", type=str, default="../model/bge-reranker-v2-m3.rknn")
    parser.add_argument("--platform", type=str, default="rk1828",
                        help="目标平台 (默认 rk1828;注意 toolkit 默认是 rk1820)")
    parser.add_argument("--seq_len", type=int, default=512, help="固定序列长度,需与 export_onnx 一致")
    parser.add_argument("--hidden", type=int, default=1024, help="hidden size (bge-reranker-v2-m3=1024)")
    parser.add_argument("--core-num", type=int, default=8, dest="core_num",
                        help="模型使用的 NPU 核数 (transformer 模型可多核;24 层单核会 MODEL_SETUP fail,"
                             "须分布到多核。板端 init_runtime 的 core_mask 须与此匹配,如 8 核=0xff)")
    parser.add_argument("--quant", action="store_true",
                        help="启用 w8a8 int8 量化(需 --dataset 指向 npy 列表;默认 fp16 不量化)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="量化校准数据集 .txt(每行一组 npy:input_ids.npy attention_mask.npy)")
    args = parser.parse_args()

    L = args.seq_len
    rknn = RKNN(verbose=True)

    print("--> config model")
    cfg = dict(target_platform=args.platform, core_num=args.core_num)
    if args.quant:
        cfg.update(quantized_dtype="w8a8", quantized_algorithm="normal", quantized_method="channel")
    rknn.config(**cfg)
    print("done")

    print("--> Loading model")
    ret = rknn.load_onnx(
        model=args.onnx_path,
        inputs=["inputs_embeds", "attention_mask", "position_ids"],
        input_size_list=[[1, L, args.hidden], [1, L], [1, L]],
    )
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    print("--> Building model")
    if args.quant:
        if not args.dataset or not os.path.exists(args.dataset):
            print("✗ --quant 需要 --dataset 指向存在的校准列表"); exit(1)
        ret = rknn.build(do_quantization=True, dataset=args.dataset)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    print("--> Export rknn model")
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print("Export rknn failed!")
        exit(ret)
    print("done")

    rknn.release()


if __name__ == "__main__":
    main()
