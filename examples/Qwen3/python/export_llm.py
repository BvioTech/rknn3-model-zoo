import os
# NOTE(violoop): 本机走代理(localhost:7897)直连 huggingface.co；hf-mirror 当前对部分文件
# 返回 308 跳回 hf.co 导致下载失败。若需镜像，设环境变量 HF_ENDPOINT 即可（不带结尾斜杠）。
if "HF_ENDPOINT" not in os.environ:
    pass  # 使用默认 https://huggingface.co
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_quantize_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

prompt = "RKLLM"
chat_context = {
    "messages":[
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": True,
    "enable_thinking": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3 llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../model/llm/Qwen3-1.7B.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='../../../datasets/llm_quant.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    ATTN_IMPL = None
    try:
        ## 使用flashattention可以极大减少grq运行显存占用，请自行安装，注意flashattention要求模型加载类型为bf16/fp16，建议使用bf16。
        import flash_attn
        ATTN_IMPL = "flash_attention_2"
        print("Use FlashAttention2")
    except Exception as e:
        print(
            f"FlashAttention2 is unavailable ({e}), "
            "falling back to standard attention."
            "It is recommended to install FlashAttention2, as standard attention requires a large amount of video memory."
    )
    kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16
    }
    if ATTN_IMPL is not None:
        kwargs["attn_implementation"] = ATTN_IMPL
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    kwargs['config'] = config
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
        
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model, tokenizer=tokenizer, device='cuda',
                                   system_prompt=None, tools=None, system_role='system', user_role='user')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm='grq', dataset=dataset)
        

        model = model.cpu()
        
    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
        os.makedirs(export_dirname)

    # Export llm to onnx
    if ATTN_IMPL is not None:
        model.config._attn_implementation = "eager"
        for layer in model.model.layers:
            layer.self_attn.config._attn_implementation = "eager"
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')