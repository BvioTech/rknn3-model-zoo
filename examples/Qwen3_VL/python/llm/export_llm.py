import os
# NOTE(violoop): 不再硬编码 hf-mirror（当前对部分文件 308 跳回 hf.co 导致下载失败）。
# 需镜像时设环境变量 HF_ENDPOINT（不带结尾斜杠）；否则默认 https://huggingface.co。
if "HF_ENDPOINT" not in os.environ:
    pass
import sys, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from transformers import masking_utils
masking_utils.sdpa_mask = masking_utils.sdpa_mask_older_torch

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_qwen3_vl_quantize_dataset
from transformers import AutoConfig # transformers==4.57.0


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel # 增加 num_logits_to_keep 输入


class LanguageModelWithLMHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.config = model.config
        self.device = model.device

        weight_t = model.lm_head.weight.detach().t().contiguous()
        self.lm_head_weight_t = torch.nn.Parameter(weight_t, requires_grad=False)

        bias = getattr(model.lm_head, "bias", None)
        if bias is not None:
            self.lm_head_bias = torch.nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        else:
            self.lm_head_bias = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deepstack_embeds0=None,
        deepstack_embeds1=None,
        deepstack_embeds2=None,
        logits_to_keep=None,
    ):
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deepstack_visual_embeds0=deepstack_embeds0,
            deepstack_visual_embeds1=deepstack_embeds1,
            deepstack_visual_embeds2=deepstack_embeds2,
            use_cache=False,
        )

        hidden_states = outputs[0]

        # logits_to_keep: shape [K], dtype int64
        logits_to_keep = logits_to_keep.to(device=hidden_states.device, dtype=torch.long)

        selected_hidden_states = torch.index_select(
            hidden_states,
            dim=1,
            index=logits_to_keep,
        )

        logits = torch.matmul(selected_hidden_states, self.lm_head_weight_t)

        return logits

prompt = "RKLLM"
chat_context = {
    "messages":[
        {
            "role": "user",
            "content": [
                {"type": "image",},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3-VL llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="./Qwen3-VL-2B-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    
    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation_autoset'], False)
    kwargs['config'] = config
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True, _attn_implementation= "eager",
        trust_remote_code=True)
    print("Loaded Qwen3-VL model weights successfully.")
    
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer


        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.model.language_model, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.model.language_model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()
        
    wrapped_model = LanguageModelWithLMHead(model)
    wrapped_model.eval()

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if export_llm_dirname and not os.path.exists(export_llm_dirname):
        print(f"create export_llm_dirname: {export_llm_dirname}")
        os.makedirs(export_llm_dirname)

    # Export llm to onnx
    args.hidden_size = config.text_config.hidden_size
    
    # 导出这个 wrapper
    causal_llm_to_onnx(wrapped_model, args)

    # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.language_model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')