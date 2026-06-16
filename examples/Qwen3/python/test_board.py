#!/usr/bin/env python3
# NOTE(violoop): 板端 (dev14, RK3576+RK18xx) 推理脚本，改写自
# rknn3-toolkit-lite/examples/Qwen3/test.py，适配 Qwen3-0.6B 并支持 --target/--prompt。
import os
import ctypes
import time
import numpy as np
from transformers import AutoTokenizer
from rknn3lite.api import RKNN3Lite, RKLLMCallback, LLMResultCallback, LLMGetEmbedCallback, LLMTokenizerCallback

VOCAB_SIZE = 151936

system_prompt  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
prompt_prefix  = "<|im_start|>user\n"
prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"

tokenizer = None
embeds_data = None
first_token = None


def result_callback(userdata, result_ptr, state):
    global tokenizer, first_token
    if not hasattr(result_callback, "accumulated_tokens"):
        result_callback.accumulated_tokens = []
        result_callback.last_output_text = ""

    def decode_safe(tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text.split('�', 1)[0] if '�' in text else text

    if state == 5:
        print("\n\nError occurred during inference")
        return 0
    if state in (2, 3, 4):
        if result_callback.accumulated_tokens:
            try:
                safe_text = decode_safe(result_callback.accumulated_tokens)
                new_part = safe_text[len(result_callback.last_output_text):]
                if new_part:
                    print(new_part, end="", flush=True)
            except Exception as e:
                print(f"\n[Decode error: {e}]", flush=True)
        result_callback.accumulated_tokens.clear()
        result_callback.last_output_text = ""
        msg = {2: "Finished", 3: "Stop", 4: "Max new token reached"}.get(state, "Unknown")
        print(f"\n\n--------------------{msg}--------------------")
        return 0
    if state == 1:
        print("\n\nWaiting for UTF-8 encoded character")
        return 0
    if state == 0:
        n = result_ptr.contents.num_tokens
        new_tokens = [result_ptr.contents.token_ids[i] for i in range(n)]
        result_callback.accumulated_tokens.extend(new_tokens)
        if first_token is None:
            first_token = time.perf_counter()
        try:
            safe_text = decode_safe(result_callback.accumulated_tokens)
            new_part = safe_text[len(result_callback.last_output_text):]
            if new_part:
                print(new_part, end="", flush=True)
                result_callback.last_output_text += new_part
        except Exception as e:
            print(f"\n[Temp decode error: {e}], waiting for more tokens", flush=True)
            return 0
    return 0


def tokenizer_callback(userdata, text_ptr, text_len, tokens_ptr, n_tokens_max):
    text = text_ptr.decode('utf-8')
    inputs = tokenizer(text, return_tensors='np', truncation=True)
    tokens = inputs['input_ids'][0][:n_tokens_max]
    n_tokens = len(tokens)
    if n_tokens <= 0:
        print(f"Tokenizer failed for {text}")
        return n_tokens
    for i in range(n_tokens):
        tokens_ptr[i] = tokens[i]
    return n_tokens


def embed_callback(userdata, tokens_ptr, num_tokens, embed, length):
    global embeds_data
    embedding_dim = embeds_data.shape[1]
    expected_len = num_tokens * embedding_dim * np.dtype(np.float16).itemsize
    if length != expected_len:
        print("invalid embed buffer")
        return -1
    dst = np.ctypeslib.as_array(
        ctypes.cast(embed, ctypes.POINTER(ctypes.c_uint16)),
        shape=(num_tokens * embedding_dim,)
    ).view(np.float16)
    tokens = [tokens_ptr[i] for i in range(num_tokens)]
    dst[:] = embeds_data[tokens].ravel()
    return 0


def printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time):
    print("\n" + "-" * 86)
    print(" %-12s  %-15s  %-8s  %-23s  %-23s" %
          ("Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"))
    print("-" * 86)
    prefill_ms = (first_token - llm_start_time) * 1000.0
    if n_prefill_tokens:
        print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
              ("Prefill", prefill_ms, n_prefill_tokens, prefill_ms / n_prefill_tokens,
               n_prefill_tokens * 1000.0 / prefill_ms))
    decode_ms = (llm_end_time - first_token) * 1000.0
    if n_decode_tokens:
        print(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f" %
              ("Generate", decode_ms, n_decode_tokens, decode_ms / n_decode_tokens,
               n_decode_tokens * 1000.0 / decode_ms))
    print("-" * 86)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Board inference for Qwen3-0.6B RKNN")
    parser.add_argument("--rknn_path", type=str, default="model/Qwen3-0.6B.rknn")
    parser.add_argument("--weight_path", type=str, default="model/Qwen3-0.6B.weight")
    parser.add_argument("--embed_path", type=str, default="model/Qwen3-0.6B.embed.bin")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument("--target", type=str, default="rk1828", help="coprocessor: rk1828 / rk1820")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prompt", type=str, nargs='*', default=["你是谁？", "请用一句话解释相对论。"])
    args = parser.parse_args()

    ARGS = [{"max_new_tokens": args.max_new_tokens,
             "top_k": 1, "top_p": 0.9, "temperature": 0.7, "repeat_penalty": 1.0,
             "vocab_size": VOCAB_SIZE, "special_eos_id": 151645,
             "max_context_len": 1024, "keep_history": 0}]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    embeds_data = np.fromfile(args.embed_path, dtype=np.float16).reshape(VOCAB_SIZE, -1)

    rknn = RKNN3Lite(llm_mode=True, verbose=True)

    print('--> Loading model')
    ret = rknn.load_rknn(args.rknn_path, args.weight_path)
    if ret != 0:
        print('Load model failed!'); exit(ret)
    print('done')

    callback = RKLLMCallback()
    callback.result_callback = LLMResultCallback(result_callback)
    callback.result_userdata = None
    callback.tokenizer_callback = LLMTokenizerCallback(tokenizer_callback)
    userdata = ctypes.py_object(tokenizer)
    callback.tokenizer_userdata = ctypes.cast(ctypes.pointer(userdata), ctypes.c_void_p)
    callback.embed_callback = LLMGetEmbedCallback(embed_callback)
    userdata2 = ctypes.py_object(embeds_data)
    callback.embed_userdata = ctypes.cast(ctypes.pointer(userdata2), ctypes.c_void_p)

    print(f'--> Init runtime environment (target={args.target})')
    ret = rknn.init_runtime(target=args.target, core_mask=0xff, llm_args=ARGS, llm_callback=callback)
    if ret != 0:
        print('Init runtime environment failed!'); exit(ret)
    print('done')

    ret = rknn.set_chat_template(system_prompt, prompt_prefix, prompt_postfix)
    if ret != 0:
        print('Set chat template failed!'); exit(ret)

    for prompt in args.prompt:
        print(f"\n==================== PROMPT ====================\n{prompt}\n--------------------------------------------------")
        ret, [n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time] = rknn.session_run(prompt=prompt)
        if ret != 0:
            print('RKNN llm inference failed!'); exit(ret)
        printf_perf(first_token, n_decode_tokens, n_prefill_tokens, llm_start_time, llm_end_time)
        first_token = None
    print('\ndone')
    rknn.release()
