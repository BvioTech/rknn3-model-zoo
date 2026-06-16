import os
# NOTE(violoop): 不再硬编码 hf-mirror（当前对部分文件 308 跳回 hf.co 导致下载失败）。
# 需镜像时设环境变量 HF_ENDPOINT（不带结尾斜杠）；否则默认 https://huggingface.co。
if "HF_ENDPOINT" not in os.environ:
    pass
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_qwen2_vl_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, Qwen3VLForConditionalGeneration # transformers==4.57.0


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3-VL vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="Qwen3-VL-2B-vision.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--img_h", type=int, help="Input image size (e.g., 384, 416, 448). Must be a multiple of 32.", required=False, default=384)
    parser.add_argument("--img_w", type=int, help="Input image size (e.g., 384, 416, 448). Must be a multiple of 32.", required=False, default=384)
    
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
         
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    kwargs['config'] = config
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, 
        low_cpu_mem_usage=True, _attn_implementation="eager", torch_dtype=torch.float32, 
        trust_remote_code=True)

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer


        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.model.visual, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.model.visual = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()
        
    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if export_vision_dirname and not os.path.exists(export_vision_dirname):
        print(f"create export_vision_dirname: {export_vision_dirname}")
        os.makedirs(export_vision_dirname)

    patch_size = 16
    if config.vision_config.patch_size is not None:
        patch_size = config.vision_config.patch_size
    else:
        print(f"patch_size is None, use default value {patch_size}")

    # export vision model
    export_qwen2_vl_vision(model.visual, args, patch_size) # 添加grid_thw对输入图片的patch
