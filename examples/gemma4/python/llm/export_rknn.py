import numpy as np
from rknn.api import RKNN

# e2b / e4b 模型路径配置
MODEL_CONFIGS = {
    'e2b': {
        'onnx_path':    '../../model/llm/gemma-4-e2b-it.onnx',
        'config_path':  '../../model/llm/gemma-4-e2b-it.config.pkl',
        'rknn_path':    '../../model/llm/gemma-4-e2b-it.rknn',
    },
    'e4b': {
        'onnx_path':    '../../model/llm/gemma-4-e4b-it.onnx',
        'config_path':  '../../model/llm/gemma-4-e4b-it.config.pkl',
        'rknn_path':    '../../model/llm/gemma-4-e4b-it.rknn',
    },
}

DATASET_PATH = None

# NOTE(violoop): 上游 V1.1.0 把这个值从 16*1024 降到了 4096。RK1828 板载 DDR 5120MB,
# E4B 权重占 ~2.9GB,16384 上下文的 KV cache 只要 ~160MB(实测 free 5071MB → 2064MB),
# 内存完全够;降到 4096 属于无谓的能力损失。这里保持 16384。
# KV cache 在 init 时按此值静态分配,与实际 prompt 长度无关。
MAX_CONTEXT_LEN = 16 * 1024
SLIDING_WINDOW  = 512

# 运行期采样参数。与 examples/gemma4/cpp/main.cc 的 SAMPLE_PARAMS 保持一致,
# 转换时一并写进 <model>.runtime.json,避免板端和转换端各自硬编码后悄悄漂移。
DEFAULT_SAMPLING = {
    'top_k'            : 1,      # 1 = greedy
    'top_p'            : 0.9,
    'temperature'      : 1.0,
    'repeat_penalty'   : 1.0,
    'frequency_penalty': 0.0,
    'presence_penalty' : 0.0,
}

def _toolkit_version():
    try:
        import rknn
        return getattr(rknn, '__version__', None) or rknn.api.rknn.__version__
    except Exception:
        try:
            from importlib.metadata import version
            return version('rknn-toolkit3')
        except Exception:
            return 'unknown'


def _hf_generation_config(model_path):
    """读 HF 的 generation_config.json,仅作参考记录(不覆盖 --top_k 等实参)。"""
    if not model_path:
        return None
    import json
    import os
    p = os.path.join(model_path, 'generation_config.json')
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            g = json.load(f)
    except Exception:
        return None
    return {k: g[k] for k in ('do_sample', 'temperature', 'top_k', 'top_p', 'eos_token_id',
                              'bos_token_id', 'pad_token_id') if k in g}


def write_runtime_json(path, args, llm_config, dynamic_input, quant):
    """把运行期契约落盘。

    板端 C++ (examples/gemma4/cpp) 目前把 max_context_len 和 SAMPLE_PARAMS 硬编码在
    main.cc 里,与转换端没有任何联系——转换时改了上下文长度,板端不改就会踩
    "max_context_len > llm_config.max_ctx_len" 直接报错,或者白白浪费已分配的 KV。
    这个文件就是两边的唯一事实来源:部署时随 .rknn 一起拷到板上,启动脚本从这里取值。
    """
    import json
    import os

    full = next(c for c in llm_config['attention_config'] if c['attention_type'] == 'FullAttention')
    sld  = next(c for c in llm_config['attention_config'] if c['attention_type'] == 'SlidingAttention')

    doc = {
        'schema': 'violoop/rknn-llm-runtime@1',
        'model': os.path.basename(os.path.splitext(args.rknn_path)[0]),
        'platform': args.platform,
        'toolkit': _toolkit_version(),

        # ---- 上下文 ----
        'context': {
            'max_context_len': args.max_context_len,
            'max_position_embeddings': args.max_context_len,
            'sliding_window': args.sliding_window,
            'prefill_chunk_len': dynamic_input[1][0][1],
            'note': 'KV cache 在 init 时按 max_context_len 静态分配,与实际 prompt 长度无关。'
                    '板端 <max_context_len> 参数必须 <= 此值,否则 demo 直接报错退出;'
                    '但它只是校验值——gemma4 cpp 把 params.max_context_len 置 0,'
                    '运行时实际总是按模型里的 max_ctx_len 分配,传小了不省内存,只会多一条 warning。',
        },

        # ---- 采样 ----
        'sampling': {k: getattr(args, k) for k in DEFAULT_SAMPLING},
        'hf_generation_config': _hf_generation_config(args.model_path),

        # ---- 量化 / KV ----
        'quantization': quant,
        'kvcache': {
            'full_attention': {k: full[k] for k in
                               ('kvcache_buffer_len', 'kvcache_dtype', 'kvcache_store_method',
                                'kvcache_group_size', 'kvcache_residual_depth')},
            'sliding_attention': {k: sld[k] for k in
                                  ('kvcache_buffer_len', 'kvcache_dtype', 'kvcache_store_method',
                                   'kvcache_group_size', 'kvcache_residual_depth')},
        },
        'rope_cache_host_storage': bool(full.get('position_embeddings_host_storage')),
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return doc


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export gemma llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, default=None, help="onnx model path")
    parser.add_argument("--config", type=str, default=None, help="config file path")
    parser.add_argument("--rknn_path", type=str, default=None, help="output rknn model path")
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument("--model_type", type=str, choices=['e2b', 'e4b'], default='e2b',
                        help="选择要导出的 Gemma-4 模型版本: e2b 或 e4b")
    parser.add_argument("--platform", type=str, default='rk1828',
                        help="目标平台 (默认 rk1828; 原脚本写死 rk1820)")
    parser.add_argument("--max_context_len", type=int, default=MAX_CONTEXT_LEN,
                        help="最大上下文长度(kvcache_buffer_len / max_position_embeddings),"
                             "KV cache 按此值静态分配 (默认 %(default)s)")
    parser.add_argument("--sliding_window", type=int, default=SLIDING_WINDOW,
                        help="滑窗层的窗口大小,与上下文长度无关 (默认 %(default)s)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="HF 模型目录/repo id,仅用于读取 generation_config.json 记进 runtime.json")
    parser.add_argument("--runtime_json", type=str, default=None,
                        help="运行期参数输出路径 (默认 <rknn_path 去掉 .rknn>.runtime.json)")
    for _k, _v in DEFAULT_SAMPLING.items():
        parser.add_argument(f"--{_k}", type=type(_v), default=_v,
                            help=f"采样参数 {_k} (默认 %(default)s)")
    args = parser.parse_args()

    # 根据 model_type 选择对应配置
    cfg = MODEL_CONFIGS[args.model_type]
    args.onnx_path = args.onnx_path or cfg['onnx_path']
    args.config    = args.config or cfg['config_path']
    args.rknn_path = args.rknn_path or cfg['rknn_path']

    import os
    # os.chdir("../../model/llm/")

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    
    from rknn.api import DEFAULT_RKNN_LLM_CONFIG
    kvcache_buffer_len = args.max_context_len
    max_position_embeddings = args.max_context_len
    my_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    sliding_window = args.sliding_window

    attn_config = {                                   # accept multi internal kvcache as list(dict)
        'mask_name'              : 'attention_mask',         # required, this is used for recognize which attn belong to this config
        'position_name'          : 'position_ids',           # allow None if no position_ids input
        'kvcache_buffer_len'     : kvcache_buffer_len,                     # feed int or str "real_time"
        'max_position_embeddings': max_position_embeddings,
        'kvcache_dtype'          : 'Int4_to_F16',                # Float16, Int8_to_F16, Int4_to_F16
        'kvcache_store_method'   : 'GroupQuant',                 # GroupQuant, Normal
        'kvcache_group_size'     : 16,
        'kvcache_residual_depth' : 64,
        'sliding_window_size'    : -1,
        'attention_type'         : 'FullAttention',          # FullAttention, SlidingAttention
        'position_embeddings_host_storage'     : True,
    }
        
    attn_sld_config = {                                   # accept multi internal kvcache as list(dict)
        'mask_name'              : 'attention_mask_1',         # required, this is used for recognize which attn belong to this config
        'position_name'          : 'position_ids_1',           # allow None if no position_ids input
        'kvcache_buffer_len'     : sliding_window,                     # feed int or str "real_time"
        'max_position_embeddings': max_position_embeddings,
        'kvcache_dtype'          : 'Float16',                # Float16, Int8_to_F16, Int4_to_F16
        'kvcache_store_method'   : 'Normal',                 # GroupQuant, Normal
        'kvcache_group_size'     : 32,
        'kvcache_residual_depth' : 32,
        'sliding_window_size'    : sliding_window,
        'attention_type'         : 'SlidingAttention',          # FullAttention, SlidingAttention
        'position_embeddings_host_storage'     : True,
    }
    attention_config = []
    attention_config.append(attn_sld_config)
    attention_config.append(attn_config)

    my_config['attention_config'] = attention_config

    print('LLM config is:', my_config)

    if "2b" in args.onnx_path.lower():
        dynamic_input = [[[1, 1], [1, 1, 35, 256], [1, 1], [1, 1], [1, 1], [1, 1], [1]], 
                            [[1, 128], [1, 128, 35, 256], [1, 128], [1, 128], [1, 128], [1, 128], [1]]]
    elif "4b" in args.onnx_path.lower():
        dynamic_input = [[[1, 1], [1, 1, 42, 256], [1, 1], [1, 1], [1, 1], [1, 1], [1]], 
                            [[1, 128], [1, 128, 42, 256], [1, 128], [1, 128], [1, 128], [1, 128], [1]]]

    quant = {'quantized_dtype': 'w4a16', 'quantized_algorithm': 'normal', 'quantized_method': 'group32'}

    rknn.config(target_platform=args.platform, dynamic_input = dynamic_input, profile_mode=False,
                llm_config=my_config,
                input_attrs={'per_layer_inputs': {'dtype': 'float16', 'layout': 'NCHW'}},
                **quant)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_llm(model=args.onnx_path, config=args.config)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=args.dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    #Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    rknn.release()

    # Export runtime contract (context length + sampling params) alongside the .rknn
    print('--> Export runtime params')
    runtime_json = args.runtime_json or (os.path.splitext(args.rknn_path)[0] + '.runtime.json')
    doc = write_runtime_json(runtime_json, args, my_config, dynamic_input, quant)
    print(f'  max_context_len   : {doc["context"]["max_context_len"]}')
    print(f'  sliding_window    : {doc["context"]["sliding_window"]}')
    print(f'  prefill_chunk_len : {doc["context"]["prefill_chunk_len"]}')
    print(f'  sampling          : {doc["sampling"]}')
    print(f'  -> {runtime_json}')
    print('done')

