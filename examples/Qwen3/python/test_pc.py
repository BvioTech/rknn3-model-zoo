#!/usr/bin/env python3
# NOTE(violoop): PC 端模拟器验证脚本，改写自 rknn3-toolkit/examples/qwen2_5/test.py。
# 模拟器推理要求经 load_llm + build（不能用 load_rknn），因此本脚本会重新量化构建。
# 本机走代理直连 HF，tokenizer 用缓存，不设 HF_ENDPOINT 镜像。
import os
from rknn.api import RKNN
import numpy as np
import torch
from transformers import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)

MODEL_DIR = '../model/llm'
ONNX_MODEL = f'{MODEL_DIR}/Qwen3-0.6B.onnx'
LLM_CONFIG = f'{MODEL_DIR}/Qwen3-0.6B.config.pkl'
RKNN_MODEL = f'{MODEL_DIR}/Qwen3-0.6B.rknn'
EMBED_PATH = f'{MODEL_DIR}/Qwen3-0.6B.embed.bin'
DATASET_PATH = '../../../datasets/CMMLU/dataset.txt'
TOKENIZER_PATH = 'Qwen/Qwen3-0.6B'
PROMPT = "请解释一下相对论的基本概念"

VOCAB_SIZE = 151936
SEQ_LEN = 128


def llm_logitsprocessor(input_ids, logits, args={}):
    temperature = args.get('temperature', 1.0)
    top_k = args.get('top_k', 1)
    top_p = args.get('top_p', 1.0)
    repetition_penalty = args.get('repeat_penalty', 1.0)
    do_sample = args.get('do_sample', False)

    warpers = [
        TemperatureLogitsWarper(temperature),
        RepetitionPenaltyLogitsProcessor(repetition_penalty) if input_ids is not None else None,
        TopKLogitsWarper(top_k=top_k),
        TopPLogitsWarper(top_p=top_p),
    ]
    for warper in warpers:
        if warper is not None:
            logits = warper(input_ids=input_ids, scores=logits)

    probs = torch.softmax(logits, dim=-1)
    if do_sample:
        next_token = torch.multinomial(probs, num_samples=1)[0]
    else:
        next_token = torch.argmax(probs, dim=-1)
    return next_token.numpy()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="PC simulator validation for Qwen3 RKNN model")
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--platform", type=str, default="rk1828")
    parser.add_argument("--target", action='store_true', help="Inference on real target instead of simulator")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    args = parser.parse_args()

    rknn = RKNN(verbose=False)

    print('--> Config model')
    rknn.config(target_platform=args.platform,
                quantized_dtype='w4a16', quantized_algorithm='grq', quantized_method='group32')
    print('done')

    print('--> Loading model')
    ret = rknn.load_llm(model=args.onnx_path, config=args.config, seq_lens=[1, SEQ_LEN])
    if ret != 0:
        print('Load model failed!'); exit(ret)
    print('done')

    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=args.dataset_path)
    if ret != 0:
        print('Build model failed!'); exit(ret)
    print('done')

    print('--> Init runtime environment')
    if args.target:
        ret = rknn.init_runtime(target=args.platform, core_mask=0xff)
    else:
        ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!'); exit(ret)
    print('done')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='np', truncation=True)

    eos_token_ids = [tokenizer.eos_token_id]
    generate_ids = []

    token_ids = inputs.input_ids
    input_seq_len = token_ids.shape[1]

    embeds_data = np.fromfile(EMBED_PATH, dtype=np.float16).reshape(VOCAB_SIZE, -1)
    embeds = embeds_data[token_ids].astype(np.float32)

    rope_cache = rknn.query('QUERY_ROPE_CACHE')

    print('--> Prefill Inference')
    if SEQ_LEN >= input_seq_len:
        inputs_embeds = np.zeros((1, SEQ_LEN, embeds.shape[-1]), dtype=np.float32)
        attention_mask = np.zeros((1, SEQ_LEN), dtype=np.float32)
        inputs_embeds[:, :input_seq_len, :] = embeds
        attention_mask[:, :input_seq_len] = 1
        num_logits_to_keep = np.array([input_seq_len - 1], dtype=np.int32)

        attention_inputs, dynamic_idx = rknn.kvcache_controller.generate_kvcache_control_tensors(input_seq_len)
        prefill_inputs = [inputs_embeds, attention_mask, num_logits_to_keep] + rope_cache + attention_inputs[0]
        data_format = ['nchw'] * len(prefill_inputs)
        prefill_logits = rknn.inference(prefill_inputs, data_format, accuracy_analysis=False)[0]
    else:
        attention_inputs, dynamic_idx = rknn.kvcache_controller.generate_kvcache_control_tensors(input_seq_len)
        for i, seq_len in enumerate(range(0, input_seq_len, SEQ_LEN)):
            inputs_embeds = np.zeros((1, SEQ_LEN, embeds.shape[-1]), dtype=np.float32)
            attention_mask = np.zeros((1, SEQ_LEN), dtype=np.float32)
            curr_len = min(SEQ_LEN, input_seq_len - seq_len)
            inputs_embeds[:, :curr_len, :] = embeds[:, seq_len:seq_len + curr_len, :]
            attention_mask[:, :curr_len] = 1
            num_logits_to_keep = np.array([curr_len - 1], dtype=np.int32)
            prefill_inputs = [inputs_embeds, attention_mask, num_logits_to_keep] + rope_cache + attention_inputs[i]
            data_format = ['nchw'] * len(prefill_inputs)
            prefill_logits = rknn.inference(prefill_inputs, data_format, accuracy_analysis=False)[0]

    next_token = llm_logitsprocessor(torch.from_numpy(token_ids), torch.from_numpy(prefill_logits).reshape(1, -1))
    print(f"--> First new token: {next_token}")
    generate_ids.append(next_token[0])

    print("--> Decoder inference")
    from tqdm import tqdm
    for i in tqdm(range(args.max_new_tokens), desc='RKLLM KVCache Inference', ncols=100):
        if i == args.max_new_tokens - 1:
            continue
        input_ids = np.expand_dims(next_token, axis=0).astype(np.int64)
        inputs_embeds = embeds_data[input_ids].astype(np.float32)
        attention_mask = np.expand_dims(np.array([1]), axis=0).astype(np.float32)
        num_logits_to_keep = np.array([0]).astype(np.int32)
        token_ids = np.concatenate((token_ids, input_ids), axis=1)

        attention_inputs, dynamic_idx = rknn.kvcache_controller.generate_kvcache_control_tensors(1)
        decoder_inputs = [inputs_embeds, attention_mask, num_logits_to_keep] + rope_cache + attention_inputs[0]
        data_format = ['nchw'] * len(decoder_inputs)
        decoder_logits = rknn.inference(decoder_inputs, data_format)[0]
        next_token = llm_logitsprocessor(torch.from_numpy(token_ids), torch.from_numpy(decoder_logits).reshape(1, -1))
        generate_ids.append(next_token[0])
        if next_token[-1] in eos_token_ids:
            print('LLM Inference has completed!')
            break

    response = tokenizer.decode(generate_ids, skip_special_tokens=True)
    print("\n==================== PROMPT ====================")
    print(args.prompt)
    print("==================== RESPONSE ====================")
    print(response)
    print("==================================================")
    rknn.release()
