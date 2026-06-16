import os
# NOTE(violoop): 不再硬编码 hf-mirror（当前对部分文件 308 跳回 hf.co 导致下载失败）。
# 需镜像时设环境变量 HF_ENDPOINT（不带结尾斜杠）；否则默认 https://huggingface.co。
if "HF_ENDPOINT" not in os.environ:
    pass
import torch
import shutil
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from PIL import Image
from onnx import numpy_helper

def load_onnx_tensor(init):
    return torch.from_numpy(numpy_helper.to_array(init))


def onnx_fingerprint(init, n=1024):
    t = load_onnx_tensor(init).reshape(-1)
    if t.numel() > n:
        t = t[:n]
    if not t.is_floating_point():
        t = t.float()
    return t.abs().sum().item()


def torch_fingerprint(t, n=1024):
    t = t.reshape(-1)
    if t.numel() > n:
        t = t[:n]
    return t.abs().sum().item()

def l1_streaming(a, b, best, block=4096):
    a = a.reshape(-1)
    b = b.reshape(-1)
    s = 0.0
    for i in range(0, a.numel(), block):
        s += (a[i:i+block] - b[i:i+block]).abs().sum().item()
        if s > best:
            return s
    return s


def replace_keys_greedy(A: dict, B: dict):
    MAP_DICT = {}
    wrong_LIST = []
    for b_key, b_val in tqdm(B.items()):
        best_sim = float("inf")
        best_a_key = None
        need_transpose = False

        # ---- 2. 遍历当前剩余的 A ----
        for a_key, a_info in A.items():
            # if 'Gather' in a_info["op_type"] and len(a_info["op_type"]) == 1:
            #     continue
            
            a_shape = a_info["shape"]
            b_shape = tuple(b_val.shape)
            match = False
            transpose = False
            if a_shape == b_shape:## 注意这里并不能保证无需转置
                match = True
            elif len(a_shape) == 2 and a_shape[::-1] == b_shape:
                match = True
                transpose = True

            if not match:
                continue
            
            a_fp = a_info["fp"]
            if len(b_shape) == 1:
                b_val_fp = torch_fingerprint(b_val)
                fingerprint_sim = abs(a_fp - b_val_fp)
                if fingerprint_sim > 1:
                    continue
            else:
                b_val_fp = torch_fingerprint(b_val)
                b_val_fp_T = torch_fingerprint(b_val.transpose(0, 1))
                fingerprint_sim = abs(a_fp - b_val_fp)
                fingerprint_sim_T = abs(a_fp - b_val_fp_T)
                if min(fingerprint_sim, fingerprint_sim_T) > 1:
                    continue
                if fingerprint_sim_T < fingerprint_sim and not transpose:
                    transpose = True
            
            a_tensor = load_onnx_tensor(a_info["init"])
            if transpose:
                a_tensor = a_tensor.transpose(0, 1)

            # ---- 3. 避免中间大 tensor ----
            sim = l1_streaming(a_tensor, b_val, best_sim)


            if sim < best_sim:
                best_sim = sim
                best_a_key = a_key
                need_transpose = transpose

            if best_sim < 0.1:
                break

        # if 'Gather' not in A[best_a_key]["op_type"]: ## 不要扔掉onnx中的embedding weight
        if best_a_key is not None:
            A.pop(best_a_key)
            
        if best_sim > 1:
            wrong_LIST.append([b_key, best_a_key, best_sim])
        else:
            # MAP_DICT[b_key] = [best_a_key, need_transpose]
            MAP_DICT[b_key] = best_a_key

        
    return MAP_DICT, wrong_LIST

def get_map(onnx_model, torch_model_dict, save_map_path, filter=None):

    onnx_dict = {}
    for init in onnx_model.graph.initializer:
        onnx_dict[init.name] = {
        "init": init,
        "shape": tuple(init.dims),
        "op_type": [],
        "fp": onnx_fingerprint(init)
    }

        
    init_to_op_type = {}
    for node in onnx_model.graph.node:
        for input_name in node.input:
            if input_name in onnx_dict:
                if input_name not in init_to_op_type:
                    init_to_op_type[input_name] = []
                init_to_op_type[input_name].append(node.op_type)
    for name, info in onnx_dict.items():
        onnx_dict[name]["op_type"] = init_to_op_type[name]
        
    if filter is not None:
        for key in list(torch_model_dict.keys()):
            if not key.startswith(filter):
                torch_model_dict.pop(key)

    MAP_DICT, wrong_LIST = replace_keys_greedy(onnx_dict, torch_model_dict)
    # if len(wrong_LIST) > 0:
    #     print('some error occurs when generate namemap dict !')
    #     print(wrong_LIST)

    with open(save_map_path, 'w', encoding='utf-8') as f:
        json.dump(
            MAP_DICT,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True
        )
    
def gen_grq_input_embeds_dataset(datasets, dataset_out_path_np, grq_data_path):
    with open(grq_data_path, 'w') as json_file:
        json_file.write('[\n')
        first = True
        for i, data in enumerate(tqdm(datasets, desc='Make grq dataset', ncols=100)):
            input_embed = np.load(os.path.join(dataset_out_path_np, f'input_embeds_{i}.npy'))
            target = data["target"]
            input_dict = {
                "input_embed": input_embed.tolist(),
                "target": target
            }
            if not first:
                json_file.write(',\n')
            else:
                first = False
            json.dump(input_dict, json_file)
        json_file.write('\n]')


def clear_llm_external_weight_in_dir(dir_path):
    remove_name = ['.weight', '.bias', '__value', 'onnx__MatMul']
    for f in os.listdir(dir_path):
        for r in remove_name:
            if r in f:
                os.remove(os.path.join(dir_path, f))
                break
    print("Done")

def gen_quantize_dataset(model_path, embed_layer, dataset_path, dataset_out_path, dataset_out_path_np):
    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)

    # if not torch.cuda.is_available():
    #     dev = 'cpu'
    # else:            
    #     dev = 'cuda'
    dev = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    datasets = json.load(open(dataset_path, 'r'))
    with open(dataset_out_path, 'w') as f:
        for i,data in enumerate(datasets):
            text = data['input'] + (data['target'] if 'target' in data else '')
            input_ids = tokenizer.encode(text, return_tensors="pt")
            input_ids = input_ids.to(dev)
            input_embeds = embed_layer(input_ids)
            input_embeds = input_embeds.float()

            n_token = input_embeds.shape[1]

            attention_mask = np.ones((1, n_token), dtype=np.float32)
            position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
            num_logits_to_keep = np.array(n_token - 1).astype(np.int32).reshape(1)

            input_embeds = input_embeds.float()
            path_input_embeds = '{}/{}_{}.npy'.format(dataset_out_path_np, 'input_embeds', i)
            np.save(path_input_embeds, input_embeds.cpu().detach().numpy())

            path_attention_mask = '{}/{}_{}.npy'.format(dataset_out_path_np, 'attention_mask', i)
            np.save(path_attention_mask, attention_mask)

            path_position_ids = '{}/{}_{}.npy'.format(dataset_out_path_np, 'position_ids', i)
            np.save(path_position_ids, position_ids)

            path_num_logits_to_keep = '{}/{}_{}.npy'.format(dataset_out_path_np, 'num_logits_to_keep', i)
            np.save(path_num_logits_to_keep, num_logits_to_keep)

            f.write(os.path.abspath(path_input_embeds) + ' '
                + os.path.abspath(path_attention_mask) + ' '
                + os.path.abspath(path_position_ids) + ' '
                + os.path.abspath(path_num_logits_to_keep) + ' ')            
            f.write('\n')

def gen_qwen3_5_quantize_dataset(model, tokenizer, embed_layer, dataset_path, dataset_out_path, dataset_out_path_np):
    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_model = model.model if hasattr(model, "model") else model
    base_model = base_model.to(dev)
    base_model.eval()
    
    # 确保 embed_layer 也在同样的设备上（如果它是独立传入的模型层）
    if hasattr(embed_layer, 'to'):
        embed_layer = embed_layer.to(dev)

    with open(dataset_path, 'r', encoding='utf-8') as fr:
        datasets = json.load(fr)

    with open(dataset_out_path, 'w', encoding='utf-8') as f:
        for i, data in enumerate(datasets):
            text = data['input'] + (data['target'] if 'target' in data else '')

            model_inputs = tokenizer([text], return_tensors="pt")
            # 将输入数据送入 CUDA
            input_ids = model_inputs['input_ids'].to(dev)

            if 'attention_mask' in model_inputs:
                attention_mask = model_inputs['attention_mask'].to(dev).to(torch.float32)
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.float32, device=dev)

            src_len = input_ids.shape[1]
            # 这里的辅助 tensor 直接在 dev 上创建
            position_ids = torch.arange(src_len, dtype=torch.long, device=dev).unsqueeze(0)
            num_logits_to_keep = torch.tensor([src_len - 1], dtype=torch.long, device=dev)
            tc_in = torch.tensor([src_len], dtype=torch.int64, device=dev)

            with torch.no_grad():
                input_embeds = embed_layer(input_ids).to(torch.float32)

            with torch.no_grad():
                model_out = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    num_logits_to_keep=num_logits_to_keep,
                    Tc=tc_in
                )

            past_key_values = model_out.past_key_values
            conv_states_list = [v for v in past_key_values.conv_states if v is not None]
            if len(conv_states_list) == 0:
                raise RuntimeError(f"sample {i}: past_key_values.conv_states is empty")

            # concat 操作由于输入是 GPU tensor，结果也会在 GPU 上
            conv_state = torch.cat(conv_states_list, dim=0).to(torch.float32)

            # 2. 保存时强制将数据脱离计算图 (detach)、转回 CPU (cpu)，再转 numpy
            path_input_embeds = os.path.join(dataset_out_path_np, f'input_embeds_{i}.npy')
            np.save(path_input_embeds, input_embeds.detach().cpu().numpy().astype(np.float32))

            path_position_ids = os.path.join(dataset_out_path_np, f'position_ids_{i}.npy')
            np.save(path_position_ids, position_ids.detach().cpu().numpy().astype(np.int64))

            path_attention_mask = os.path.join(dataset_out_path_np, f'attention_mask_{i}.npy')
            np.save(path_attention_mask, attention_mask.detach().cpu().numpy().astype(np.float32))

            path_linear_tc = os.path.join(dataset_out_path_np, f'Linear_Tc_{i}.npy')
            np.save(path_linear_tc, tc_in.detach().cpu().numpy().astype(np.int64))

            path_num_logits_to_keep = os.path.join(dataset_out_path_np, f'num_logits_to_keep_{i}.npy')
            np.save(path_num_logits_to_keep, num_logits_to_keep.detach().cpu().numpy().astype(np.int64))

            path_conv_state = os.path.join(dataset_out_path_np, f'conv_state_{i}.npy')
            np.save(path_conv_state, conv_state.detach().cpu().numpy().astype(np.float32))

            f.write(
                os.path.abspath(path_input_embeds) + ' ' +
                os.path.abspath(path_position_ids) + ' ' +
                os.path.abspath(path_attention_mask) + ' ' +
                os.path.abspath(path_linear_tc) + ' ' +
                os.path.abspath(path_num_logits_to_keep) + ' ' +
                os.path.abspath(path_conv_state)
            )
            f.write('\n')



def gen_smolvlm_quantize_dataset(model_path, model, embed_layer, dataset_path, dataset_out_path, dataset_out_path_np, grq_data=False):
    import os
    import shutil
    import json
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor

    class smolvl_vison(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vision_model = model.model.vision_model
            self.connector = model.model.connector
            self.config = model.model.config
            self.pixel_shape = [1, 3, 512, 512]

        def forward(
            self,
            pixel_values: torch.FloatTensor,
        ) -> torch.FloatTensor:
        
            pixel_attention_mask = torch.ones(
                size=(self.pixel_shape[0], self.pixel_shape[2], self.pixel_shape[3]),
                dtype=torch.bool,
                device=pixel_values.device,
            )

            # 构建 patch-level 的 attention mask
            patch_size = 16
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # 视觉编码器前向传播
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            )
            image_features = vision_outputs.last_hidden_state  # [N, L_vision, D_vision]
            import numpy as np
            np.save("image_features.npy", image_features.cpu().detach().numpy())
            # 通过 connector 投影到文本嵌入空间
            image_hidden_states = self.connector(image_features)  # [N, L_text, D_text]

            return image_hidden_states

    
    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)
    
    dev = model.device # 使用模型的设备
    
    IMAGE_SIZE = 512 
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    try:
        datasets = json.load(open(dataset_path, 'r'))
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        return

    model.eval()
    vision_model = smolvl_vison(model)

    with open(dataset_out_path, 'w') as f:
        for i, data in enumerate(tqdm(datasets, desc='Make dataset', ncols=100)):
            try:
                # 获取图片路径
                image_dir = data.get("image_path", "")
                image_name = data["image"]
                imgp = os.path.join(os.path.dirname(dataset_path), image_dir, image_name)
                
                # 加载图片并统一 resize 到 512x512
                image = Image.open(imgp).convert("RGB")
                # 统一 resize 到 512x512
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
                
                # 构造对话
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": data["input"]},
                        ],
                    }
                ]
                
                text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                
                inputs = processor(
                    text=[text_prompt], 
                    images=[image], 
                    padding=True, 
                    return_tensors="pt", 
                    size={"longest_edge": IMAGE_SIZE} 
                )
                
                inputs = inputs.to(dev)
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"] # [1, 3, 512, 512] (假设预处理结果仍是 512x512)
                
                # --- 6. 获取和拼接嵌入 ---
                input_embeds = embed_layer(input_ids)
                image_mask = input_ids == model.config.image_token_id 

                image_embeds = vision_model.forward(pixel_values[0])
                
                # 插入图像嵌入（逻辑与上个版本保持一致）
                L_img_seq = image_embeds.shape[1]
                if image_mask.any():
                    first_image_token_idx = (image_mask.squeeze(0).nonzero(as_tuple=True)[0][0]).item()
                    # 替换对应的嵌入块
                    input_embeds[0, first_image_token_idx : first_image_token_idx + L_img_seq] = image_embeds.squeeze(0)
                else:
                    # 如果没有找到 <image> token，这通常表示对话模板或数据有问题
                    print(f"Warning: <image> token not found for dataset index {i}. Skipping image embedding.")


                input_embeds = input_embeds.float() # 量化通常需要 float32
                
                n_token = input_embeds.shape[1]
                attention_mask = np.ones((1, n_token), dtype=np.float32)
                position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
                num_logits_to_keep = np.array(n_token - 1).astype(np.int32).reshape(1)

                path_input_embeds = os.path.join(dataset_out_path_np, f'input_embeds_{i}.npy')
                np.save(path_input_embeds, input_embeds.cpu().detach().numpy())
                
                path_attention_mask = os.path.join(dataset_out_path_np, f'attention_mask_{i}.npy')
                np.save(path_attention_mask, attention_mask)
                
                path_position_ids = os.path.join(dataset_out_path_np, f'position_ids_{i}.npy')
                np.save(path_position_ids, position_ids)
                
                path_num_logits_to_keep = os.path.join(dataset_out_path_np, f'num_logits_to_keep_{i}.npy')
                np.save(path_num_logits_to_keep, num_logits_to_keep)
                
                # --- 8. 写入输出文件 ---
                f.write(
                    os.path.abspath(path_input_embeds) + ' ' + 
                    os.path.abspath(path_attention_mask) + ' ' + 
                    os.path.abspath(path_position_ids) + ' ' + 
                    os.path.abspath(path_num_logits_to_keep) + ' '
                )
                f.write('\n')
                
            except Exception as e:
                # 捕获之前提到的 "too many values to unpack" 或其他错误
                print(f"Skipping dataset index {i} due to error: {e}")
                continue

    print(f"SmolVLM quantization dataset generated and saved to {dataset_out_path_np}")

def gen_qwen_vl_vision_prune_quantize_dataset(dataset_path, dataset_out_path, img_h, img_w, mean, std, patch_size):
    import cv2,os
    base_dir = os.path.dirname(os.path.abspath(dataset_path))
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            img_path = line.strip()
            abs_img_path = os.path.join(base_dir, img_path) if not os.path.isabs(img_path) else img_path
            if abs_img_path.lower().endswith('.npy'):
                print("The prune model does not support npy input.")
                return -1
            dataset.append(abs_img_path)

    out_dir = os.path.dirname(dataset_out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    os.makedirs(os.path.join(out_dir, "npy"), exist_ok=True)
    
    output_paths = []
    for img_path in dataset:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_w, img_h)).transpose(2,0,1).reshape(1,3,img_h, img_w)
        img = np.float32(img)
        img[0,2,...] = (img[0,2,...] - mean[2])/std[2] 
        img[0,1,...] = (img[0,1,...] - mean[1])/std[1] 
        img[0,0,...] = (img[0,0,...] - mean[0])/std[0] 
        patches = np.concatenate([img, img], axis=1)
        h = img.shape[2]
        w = img.shape[3]
        patches = patches.reshape(1, 2, 3, h // 2 // patch_size, 2, patch_size, w // 2 // patch_size, 2, patch_size)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        feature = patches.reshape(1 * h // patch_size * w // patch_size, 3 * 2 * patch_size * patch_size)
        
        name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(out_dir, f"npy/{name_without_ext}.npy")
        np.save(str(output_path), feature)
        output_paths.append(os.path.abspath(output_path))
    
    with open(dataset_out_path, 'w', encoding='utf-8') as outfile:
        for output_path in output_paths:
            outfile.write(output_path + '\n')

    return 0


def gen_qwen2_5_vl_quantize_dataset(model_path, model, embed_layer, dataset_path, dataset_out_path, dataset_out_path_np, grq_data=False):
    from transformers import AutoProcessor

    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)

    # if not torch.cuda.is_available():
    #     dev = 'cpu'
    # else:            
    #     dev = 'cuda'
    dev = 'cpu'

    processor = AutoProcessor.from_pretrained(model_path)

    datasets = json.load(open(dataset_path, 'r'))
    with open(dataset_out_path, 'w') as f:
        for i,data in enumerate(tqdm(datasets, desc='Make dataset', ncols=100)):
            image_name = data["image"].split(".")[0]
            imgp = os.path.join(os.path.dirname(dataset_path), data["image_path"], data["image"])
            image = Image.open(imgp)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": data["input"]},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(dev)
            input_embeds = embed_layer(inputs["input_ids"])
            pixel_values = inputs["pixel_values"]
            image_mask = inputs["input_ids"] == model.config.image_token_id
            image_embeds = model.visual(pixel_values, grid_thw=inputs["image_grid_thw"])
            input_embeds[image_mask] = image_embeds
            input_embeds = input_embeds.float()

            n_token = input_embeds.shape[1]

            attention_mask = np.ones((1, n_token), dtype=np.float32)
            position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
            num_logits_to_keep = np.array(n_token - 1).astype(np.int32).reshape(1)

            path_input_embeds = '{}/{}_{}.npy'.format(dataset_out_path_np, 'input_embeds', i)
            np.save(path_input_embeds, input_embeds.cpu().detach().numpy())

            path_attention_mask = '{}/{}_{}.npy'.format(dataset_out_path_np, 'attention_mask', i)
            np.save(path_attention_mask, attention_mask)

            path_position_ids = '{}/{}_{}.npy'.format(dataset_out_path_np, 'position_ids', i)
            np.save(path_position_ids, position_ids)

            path_num_logits_to_keep = '{}/{}_{}.npy'.format(dataset_out_path_np, 'num_logits_to_keep', i)
            np.save(path_num_logits_to_keep, num_logits_to_keep)

            f.write(os.path.abspath(path_input_embeds) + ' '
                + os.path.abspath(path_attention_mask) + ' '
                + os.path.abspath(path_position_ids) + ' '
                + os.path.abspath(path_num_logits_to_keep) + ' ')            
            f.write('\n')

    if grq_data:
        grq_data_path = '../../../../datasets/MMBench/llm/grq_inputs.json'
        gen_grq_input_embeds_dataset(datasets, dataset_out_path_np, grq_data_path)
    
def gen_qwen3_5_vl_quantize_dataset(
    model_path,
    model,
    embed_layer,
    dataset_path,
    dataset_out_path,
    dataset_out_path_np,
):
    import os
    import json
    import shutil
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np, exist_ok=True)

    out_dir = os.path.dirname(dataset_out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dev = "cpu"

    model = model.to(dev).eval()
    embed_layer = embed_layer.to(dev).eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 只从 VLM 里取 visual / image_token_id / get_rope_index
    vlm = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        _attn_implementation="eager",
        trust_remote_code=True,
    ).to(dev).eval()

    visual = vlm.model.visual
    image_token_id = vlm.config.image_token_id

    datasets = json.load(open(dataset_path, "r", encoding="utf-8"))

    with open(dataset_out_path, "w", encoding="utf-8") as f:
        for i, data in enumerate(tqdm(datasets, desc="Make VLM dataset", ncols=100)):
            img_path = os.path.join(
                os.path.dirname(dataset_path),
                data["image_path"],
                data["image"],
            )
            image = Image.open(img_path).convert("RGB")

            text = data["input"] + data.get("target", "")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                }
            ]

            prompt = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(dev)

            input_ids = inputs["input_ids"]
            attention_mask_i64 = inputs["attention_mask"]
            attention_mask = attention_mask_i64.to(torch.float32)
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]

            with torch.no_grad():
                input_embeds = embed_layer(input_ids).to(torch.float32)

                image_embeds = visual(
                    pixel_values,
                    grid_thw=image_grid_thw,
                    return_dict=True,
                ).pooler_output.to(torch.float32)

                image_mask = input_ids == image_token_id
                input_embeds[image_mask] = image_embeds.reshape(-1, input_embeds.shape[-1])

                mm_token_type_ids = inputs.get("mm_token_type_ids", None)
                if mm_token_type_ids is None:
                    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
                    mm_token_type_ids[image_mask] = 1

                # 注意：get_rope_index 在 vlm.model 上，不在 vlm 上
                position_ids = vlm.model.get_rope_index(
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask_i64,
                )[0].to(torch.int64)

                seq_len = input_ids.shape[1]
                tc_in = torch.tensor([seq_len], dtype=torch.int64, device=dev)
                num_logits_to_keep = torch.tensor([seq_len - 1], dtype=torch.long, device=dev)

                model_out = model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    Tc=tc_in,
                    logits_to_keep=num_logits_to_keep,
                    use_cache=True,
                )

                conv_state = torch.cat(
                    [v for v in model_out.past_key_values.conv_states if v is not None],
                    dim=0,
                ).to(torch.float32)

            paths = [
                os.path.join(dataset_out_path_np, f"input_embeds_{i}.npy"),
                os.path.join(dataset_out_path_np, f"position_ids_{i}.npy"),
                os.path.join(dataset_out_path_np, f"attention_mask_{i}.npy"),
                os.path.join(dataset_out_path_np, f"Linear_Tc_{i}.npy"),
                os.path.join(dataset_out_path_np, f"num_logits_to_keep_{i}.npy"),
                os.path.join(dataset_out_path_np, f"conv_state_{i}.npy"),
            ]

            n_token = position_ids.shape[-1]
            position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
            arrays = [
                input_embeds.cpu().numpy().astype(np.float32),
                position_ids,
                attention_mask.cpu().numpy().astype(np.float32),
                tc_in.cpu().numpy().astype(np.int64),
                num_logits_to_keep.cpu().numpy().astype(np.int64),
                conv_state.cpu().numpy().astype(np.float32),
            ]

            for p, arr in zip(paths, arrays):
                np.save(p, arr)

            f.write(" ".join(os.path.abspath(p) for p in paths) + "\n")

def gen_qwen2_vl_quantize_dataset(model_path, model, embed_layer, dataset_path, dataset_out_path, dataset_out_path_np, grq_data=False):
    from transformers import AutoProcessor

    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)

    dev = 'cpu'
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_path, size={"shortest_edge": min_pixels, "longest_edge": max_pixels})

    datasets = json.load(open(dataset_path, 'r'))
    with open(dataset_out_path, 'w') as f:
        for i,data in enumerate(tqdm(datasets, desc='Make dataset', ncols=100)):
            image_name = data["image"].split(".")[0]
            imgp = os.path.join(os.path.dirname(dataset_path), data["image_path"], data["image"])
            image = Image.open(imgp)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": data["input"]},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(dev)
            input_embeds = embed_layer(inputs["input_ids"])
            pixel_values = inputs["pixel_values"]
            image_mask = inputs["input_ids"] == model.config.image_token_id
            image_embeds = model.visual(pixel_values, grid_thw=inputs["image_grid_thw"])
            input_embeds[image_mask] = image_embeds
            input_embeds = input_embeds.float()

            n_token = input_embeds.shape[1]

            attention_mask = np.ones((1, n_token), dtype=np.float32)
            position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
            num_logits_to_keep = np.array(n_token - 1).astype(np.int32).reshape(1)

            path_input_embeds = '{}/{}_{}.npy'.format(dataset_out_path_np, 'input_embeds', i)
            np.save(path_input_embeds, input_embeds.cpu().detach().numpy())

            path_attention_mask = '{}/{}_{}.npy'.format(dataset_out_path_np, 'attention_mask', i)
            np.save(path_attention_mask, attention_mask)

            path_position_ids = '{}/{}_{}.npy'.format(dataset_out_path_np, 'position_ids', i)
            np.save(path_position_ids, position_ids)

            path_num_logits_to_keep = '{}/{}_{}.npy'.format(dataset_out_path_np, 'num_logits_to_keep', i)
            np.save(path_num_logits_to_keep, num_logits_to_keep)

            f.write(os.path.abspath(path_input_embeds) + ' '
                + os.path.abspath(path_attention_mask) + ' '
                + os.path.abspath(path_position_ids) + ' '
                + os.path.abspath(path_num_logits_to_keep) + ' ')            
            f.write('\n')

    if grq_data:
        grq_data_path = '../../../../datasets/MMBench/llm/grq_inputs.json'
        gen_grq_input_embeds_dataset(datasets, dataset_out_path_np, grq_data_path)


def gen_qwen2_5_omni_quantize_dataset(model_path, model, embed_layer, dataset_path, dataset_out_path, dataset_out_path_np, grq_data=False):
    from transformers import Qwen2_5OmniProcessor

    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)

    # if not torch.cuda.is_available():
    #     dev = 'cpu'
    # else:            
    #     dev = 'cuda'
    dev = 'cpu'

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    datasets = json.load(open(dataset_path, 'r'))
    with open(dataset_out_path, 'w') as f:
        for i,data in enumerate(tqdm(datasets, desc='Make dataset', ncols=100)):
            image_name = data["image"].split(".")[0]
            imgp = os.path.join(os.path.dirname(dataset_path), data["image_path"], data["image"])
            image = Image.open(imgp)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": data["input"]},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=text_prompt, images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(dev)
            input_embeds = embed_layer(inputs["input_ids"])
            pixel_values = inputs["pixel_values"]
            image_mask = inputs["input_ids"] == model.config.image_token_index
            image_embeds = model.visual(pixel_values, grid_thw=inputs["image_grid_thw"])
            input_embeds[image_mask] = image_embeds
            input_embeds = input_embeds.float()

            n_token = input_embeds.shape[1]

            attention_mask = np.ones((1, n_token), dtype=np.float32)
            position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
            num_logits_to_keep = np.array(n_token - 1).astype(np.int32).reshape(1)

            path_input_embeds = '{}/{}_{}.npy'.format(dataset_out_path_np, 'input_embeds', i)
            np.save(path_input_embeds, input_embeds.cpu().detach().numpy())

            path_attention_mask = '{}/{}_{}.npy'.format(dataset_out_path_np, 'attention_mask', i)
            np.save(path_attention_mask, attention_mask)

            path_position_ids = '{}/{}_{}.npy'.format(dataset_out_path_np, 'position_ids', i)
            np.save(path_position_ids, position_ids)

            path_num_logits_to_keep = '{}/{}_{}.npy'.format(dataset_out_path_np, 'num_logits_to_keep', i)
            np.save(path_num_logits_to_keep, num_logits_to_keep)

            f.write(os.path.abspath(path_input_embeds) + ' '
                + os.path.abspath(path_attention_mask) + ' '
                + os.path.abspath(path_position_ids) + ' '
                + os.path.abspath(path_num_logits_to_keep) + ' ')            
            f.write('\n')

    if grq_data:
        grq_data_path = '../../../../datasets/MMBench/llm/grq_inputs.json'
        gen_grq_input_embeds_dataset(datasets, dataset_out_path_np, grq_data_path)


def gen_qwen3_vl_quantize_dataset(model_path, model, embed_layer, dataset_path, dataset_out_path, dataset_out_path_np, grq_data=False):
    from transformers import AutoProcessor

    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)

    # if not torch.cuda.is_available():
    #     dev = 'cpu'
    # else:            
    #     dev = 'cuda'
    dev = 'cpu'

    processor = AutoProcessor.from_pretrained(model_path)

    datasets = json.load(open(dataset_path, 'r'))
    with open(dataset_out_path, 'w') as f:
        for i,data in enumerate(tqdm(datasets, desc='Make dataset', ncols=100)):
            image_name = data["image"].split(".")[0]
            imgp = os.path.join(os.path.dirname(dataset_path), data["image_path"], data["image"])
            image = Image.open(imgp)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": data["input"]},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(dev)
            input_embeds = embed_layer(inputs["input_ids"])
            pixel_values = inputs["pixel_values"]
            image_mask = inputs["input_ids"] == model.config.image_token_id
            image_embeds, deepstack_image_embeds = model.visual(pixel_values, grid_thw=inputs["image_grid_thw"])
            input_embeds[image_mask] = image_embeds
            
            deepstack_embeds0 = torch.zeros(1, input_embeds.shape[1], input_embeds.shape[2], dtype=input_embeds.dtype, device=input_embeds.device)
            deepstack_embeds1 = torch.zeros(1, input_embeds.shape[1], input_embeds.shape[2], dtype=input_embeds.dtype, device=input_embeds.device)
            deepstack_embeds2 = torch.zeros(1, input_embeds.shape[1], input_embeds.shape[2], dtype=input_embeds.dtype, device=input_embeds.device)

            deepstack_embeds0[image_mask] = deepstack_image_embeds[0]
            deepstack_embeds1[image_mask] = deepstack_image_embeds[1]
            deepstack_embeds2[image_mask] = deepstack_image_embeds[2]
            input_embeds = input_embeds.float()
            deepstack_embeds0 = deepstack_embeds0.float()
            deepstack_embeds1 = deepstack_embeds1.float()
            deepstack_embeds2 = deepstack_embeds2.float()

            n_token = input_embeds.shape[1]

            attention_mask = np.ones((1, n_token), dtype=np.float32)
            position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
            num_logits_to_keep = np.array(n_token - 1).astype(np.int32).reshape(1)

            path_input_embeds = '{}/{}_{}.npy'.format(dataset_out_path_np, 'input_embeds', i)
            np.save(path_input_embeds, input_embeds.cpu().detach().numpy())

            path_attention_mask = '{}/{}_{}.npy'.format(dataset_out_path_np, 'attention_mask', i)
            np.save(path_attention_mask, attention_mask)

            path_position_ids = '{}/{}_{}.npy'.format(dataset_out_path_np, 'position_ids', i)
            np.save(path_position_ids, position_ids)

            path_deepstack_embeds0 = '{}/{}_{}.npy'.format(dataset_out_path_np, 'deepstack_embeds0', i)
            np.save(path_deepstack_embeds0, deepstack_embeds0.cpu().detach().numpy())

            path_deepstack_embeds1 = '{}/{}_{}.npy'.format(dataset_out_path_np, 'deepstack_embeds1', i)
            np.save(path_deepstack_embeds1, deepstack_embeds1.cpu().detach().numpy())

            path_deepstack_embeds2 = '{}/{}_{}.npy'.format(dataset_out_path_np, 'deepstack_embeds2', i)
            np.save(path_deepstack_embeds2, deepstack_embeds2.cpu().detach().numpy())

            path_num_logits_to_keep = '{}/{}_{}.npy'.format(dataset_out_path_np, 'num_logits_to_keep', i)
            np.save(path_num_logits_to_keep, num_logits_to_keep)

            f.write(os.path.abspath(path_input_embeds) + ' '+os.path.abspath(path_attention_mask) + ' '
                    + os.path.abspath(path_position_ids) + ' '
                    + os.path.abspath(path_deepstack_embeds0) + ' '
                    + os.path.abspath(path_deepstack_embeds1) + ' '
                    + os.path.abspath(path_deepstack_embeds2) + ' '
                    + os.path.abspath(path_num_logits_to_keep) + ' '
                    )            
            f.write('\n')

    if grq_data:
        grq_data_path = '../../../../datasets/MMBench/llm/grq_inputs.json'
        gen_grq_input_embeds_dataset(datasets, dataset_out_path_np, grq_data_path)



def gen_internvl_quantize_dataset(model_path, model, tokenizer,
                                  dataset_path, dataset_out_path, dataset_out_path_np,
                                  grq_data=False, image_input_size=448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width  = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(image_file, input_size=448, max_num=1):
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        # 关键：只取 1 张，且不加缩略图 —— 保持与下游 num_patches_list=1 一致
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, min_num=1, max_num=1)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)   # [1, 3, H, W]
        return pixel_values

    class ConvTemplate:
        def __init__(
            self,
            system_template="<|im_start|>system\n{system_message}",
            system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            sep="<|im_end|>\n",
        ):
            self.system_template = system_template
            self.system_message = system_message
            self.roles = roles
            self.sep = sep
            self.messages = []  # _build_internvl_query 里会直接 append_message

        def append_message(self, role_prefix: str, content):
            self.messages.append([role_prefix, content])

        def get_prompt(self) -> str:
            # 只实现 MPT 拼接：system + sep，然后每条 (role_prefix + content + sep)
            prompt = self.system_template.format(system_message=self.system_message) + self.sep
            for role_prefix, content in self.messages:
                if content is None:
                    prompt += role_prefix            # 轮到模型生成，最后一条为 None
                else:
                    prompt += role_prefix + content + self.sep
            return prompt

    def make_internvl2_5_conv():
        return ConvTemplate(
            system_template="<|im_start|>system\n{system_message}",
            system_message="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            sep="<|im_end|>\n",
        )

    def _build_internvl_query(model, tokenizer, user_text):
        """
        用 InternVL 官方同款方式拿会话模板
        然后把 <image> 替换为 <img> + <IMG_CONTEXT>*num_image_token + </img>
        """
        # 1) 正确创建模板（与 chat() 一致）
        template = make_internvl2_5_conv()
        template.system_message = model.system_message

        # 保险：某些版本下 messages 可能是 tuple，这里转成 list
        if isinstance(getattr(template, "messages", None), tuple):
            template.messages = list(template.messages)

        # 2) 构造问句
        question = f"<image>\n{user_text}"

        # 3) 进模板
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)

        # 4) 得到 query
        query = template.get_prompt()

        # 5) 把 <image> 展开为 IMG 占位 token 串
        IMG_START_TOKEN   = '<img>'
        IMG_END_TOKEN     = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * model.num_image_token) + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        return query, eos_token_id, IMG_CONTEXT_TOKEN

    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np, exist_ok=True)

    dev = 'cpu'  # 与原实现一致
    datasets = json.load(open(dataset_path, 'r'))

    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')

    with open(dataset_out_path, 'w') as f:
        for i, data in enumerate(tqdm(datasets, desc='Make dataset (InternVL)', ncols=100)):
            # 1) 读图并做预处理
            imgp = os.path.join(os.path.dirname(dataset_path), data["image_path"], data["image"])
            pixel_values = load_image(imgp, input_size=image_input_size).to(torch.float32)  # [1, 3, H, W]

            # 2) 构造 query（系统提示与模板对齐）
            user_text = data["input"]
            query, eos_token_id, IMG_CONTEXT_TOKEN = _build_internvl_query(model, tokenizer, user_text)

            # 3) tokenizer -> input_ids/attention_mask
            model_inputs = tokenizer(query, return_tensors='pt')
            input_ids = model_inputs['input_ids'].to(dev)          # [1, N]
            attention_mask = model_inputs['attention_mask'].to(dev)  # [1, N]

            # 4) 视觉特征
            with torch.no_grad():
                vit_embeds = model.extract_feature(pixel_values)   # [1, num_image_token, hidden]
                # 语言模型 token embedding
                input_embeds = model.language_model.get_input_embeddings()(input_ids)  # [1, N, hidden]

                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)
                flat_ids = input_ids.reshape(B * N)
                selected = (flat_ids == img_context_token_id)
                assert selected.sum() != 0, "未在 input_ids 中找到 IMG_CONTEXT_TOKEN！"

                # 把 IMG_CONTEXT 的位置替换为视觉特征
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
                input_embeds = input_embeds.reshape(B, N, C).float()

            # 5) 其余张量
            n_token = input_ids.shape[1]
            position_ids = torch.arange(0, n_token, dtype=torch.long).unsqueeze(0)   # [1, N]
            num_logits_to_keep = torch.tensor(n_token - 1, dtype=torch.long)         # 与原逻辑一致

            # 6) 落盘
            inputs_ = [input_embeds, attention_mask.float(), position_ids, num_logits_to_keep]
            names   = ['input_embeds', 'attention_mask', 'position_ids', 'num_logits_to_keep']
            for tensor, name in zip(inputs_, names):
                path_ = f'{dataset_out_path_np}/{name}_{i}.npy'
                np.save(path_, tensor.cpu().detach().numpy())
                f.write(os.path.abspath(path_) + ' ')
            f.write('\n')

    if grq_data:
        grq_data_path = '../../../../datasets/MMBench/llm/grq_inputs.json'
        gen_grq_input_embeds_dataset(datasets, dataset_out_path_np, grq_data_path)


def make_dataset_for_minicpm_v(path, dataset_path, dataset_out_path, dataset_out_path_np, grq_data=False):
    from transformers import AutoModel, AutoProcessor
    from copy import deepcopy

    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)

    model = AutoModel.from_pretrained(path, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    datasets = json.load(open(dataset_path, 'r'))
    with open(dataset_out_path, 'w') as f:
        for i,data in enumerate(tqdm(datasets, desc='Make dataset', ncols=100)):
            imgp = os.path.join(os.path.dirname(dataset_path), data["image_path"], data["image"])
            image = Image.open(imgp).convert('RGB')
            question = data["input"]
            msgs = [{'role': 'user', 'content': [image, question]}]

            msgs_list = [msgs]
            images_list = [None]
            prompts_lists = []
            input_images_lists = []
            for image, msgs in zip(images_list, msgs_list):
                if isinstance(msgs, str):
                    msgs = json.loads(msgs)
                copy_msgs = deepcopy(msgs)
                images = []
                for j, msg in enumerate(copy_msgs):
                    role = msg["role"]
                    content = msg["content"]
                    assert role in ["user", "assistant"]
                    if j == 0:
                        assert role == "user", "The role of first msg should be user"
                    if isinstance(content, str):
                        content = [content]
                    cur_msgs = []
                    for c in content:
                        if isinstance(c, Image.Image):
                            images.append(c)
                            cur_msgs.append("(<image>./</image>)")
                        elif isinstance(c, str):
                            cur_msgs.append(c)
                    msg["content"] = "\n".join(cur_msgs)
                prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
                input_images_lists.append(images)
            inputs = processor(
                prompts_lists, 
                input_images_lists, 
                max_slice_nums=None,
                use_image_id=None,
                return_tensors="pt", 
                max_length=8192
            ).to(model.device)
            inputs.pop("image_sizes")
            model_inputs = {
                "input_ids": inputs['input_ids'],
                "image_bound": inputs['image_bound'],
            }
            model_inputs["pixel_values"] = inputs['pixel_values']
            model_inputs['tgt_sizes'] = inputs['tgt_sizes']

            with torch.inference_mode():
                (
                    input_embeds,
                    vision_hidden_states,
                ) = model.get_vllm_embedding(model_inputs)

            n_token = input_embeds.shape[1]

            attention_mask = np.ones((1, n_token), dtype=np.float32)
            position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
            num_logits_to_keep = np.array(n_token - 1).astype(np.int32).reshape(1)

            input_embeds = input_embeds.float()
            path_input_embeds = '{}/{}_{}.npy'.format(dataset_out_path_np, 'input_embeds', i)
            np.save(path_input_embeds, input_embeds.cpu().detach().numpy())

            path_attention_mask = '{}/{}_{}.npy'.format(dataset_out_path_np, 'attention_mask', i)
            np.save(path_attention_mask, attention_mask)

            path_position_ids = '{}/{}_{}.npy'.format(dataset_out_path_np, 'position_ids', i)
            np.save(path_position_ids, position_ids)

            path_num_logits_to_keep = '{}/{}_{}.npy'.format(dataset_out_path_np, 'num_logits_to_keep', i)
            np.save(path_num_logits_to_keep, num_logits_to_keep)

            f.write(os.path.abspath(path_input_embeds) + ' '
                + os.path.abspath(path_attention_mask) + ' '
                + os.path.abspath(path_position_ids) + ' '
                + os.path.abspath(path_num_logits_to_keep) + ' ')            
            f.write('\n')

    if grq_data:
        grq_data_path = '../../../../datasets/MMBench/llm/grq_inputs.json'
        gen_grq_input_embeds_dataset(datasets, dataset_out_path_np, grq_data_path)

def make_dataset_for_fastvlm(model_path, dataset_path, dataset_out_path, dataset_out_path_np, grq_data=False):
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../examples/FastVLM/python')))
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

    IMAGE_TOKEN_INDEX = 151646

    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device="cuda")

    image_processor.size["shortest_edge"] = 512
    image_processor.crop_size = {'height': 512, 'width': 512}

    datasets = json.load(open(dataset_path, 'r'))
    with open(dataset_out_path, 'w') as f:
        for i,data in enumerate(tqdm(datasets, desc='Make dataset', ncols=100)):
            imgp = os.path.join(os.path.dirname(dataset_path), data["image_path"], data["image"])
            image = Image.open(imgp).convert('RGB')
            question = data["input"]

            question = "<|im_start|>" + "<|image_pad|>" + "<|im_end|>" + '\n' + question
            conv = conv_templates["qwen_2"].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            model.generation_config.pad_token_id = tokenizer.pad_token_id

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch.device("cuda"))

            image_tensor = process_images([image], image_processor, model.config)[0]

            inputs = model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image.size],
                labels=None,
            )

            input_embeds = inputs[4].float()

            n_token = input_embeds.shape[1]

            attention_mask = np.ones((1, n_token), dtype=np.float32)
            position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
            num_logits_to_keep = np.array(n_token - 1).astype(np.int32).reshape(1)

            input_embeds = input_embeds.float()
            path_input_embeds = '{}/{}_{}.npy'.format(dataset_out_path_np, 'input_embeds', i)
            np.save(path_input_embeds, input_embeds.cpu().detach().numpy())

            path_attention_mask = '{}/{}_{}.npy'.format(dataset_out_path_np, 'attention_mask', i)
            np.save(path_attention_mask, attention_mask)

            path_position_ids = '{}/{}_{}.npy'.format(dataset_out_path_np, 'position_ids', i)
            np.save(path_position_ids, position_ids)

            path_num_logits_to_keep = '{}/{}_{}.npy'.format(dataset_out_path_np, 'num_logits_to_keep', i)
            np.save(path_num_logits_to_keep, num_logits_to_keep)

            f.write(os.path.abspath(path_input_embeds) + ' '
                + os.path.abspath(path_attention_mask) + ' '
                + os.path.abspath(path_position_ids) + ' '
                + os.path.abspath(path_num_logits_to_keep) + ' ')            
            f.write('\n')

    if grq_data:
        grq_data_path = '../../../../datasets/MMBench/llm/grq_inputs.json'
        gen_grq_input_embeds_dataset(datasets, dataset_out_path_np, grq_data_path)

def make_dataset_for_janus_pro(model_path, dataset_path, dataset_out_path, dataset_out_path_np, grq_data=False):
    from transformers import AutoModelForCausalLM
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images

    if os.path.exists(dataset_out_path_np):
        shutil.rmtree(dataset_out_path_np)
    os.makedirs(dataset_out_path_np)

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.eval()
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

    datasets = json.load(open(dataset_path, 'r'))
    with open(dataset_out_path, 'w') as f:
        for i,data in enumerate(tqdm(datasets, desc='Make dataset', ncols=100)):
            imgp = os.path.join(os.path.dirname(dataset_path), data["image_path"], data["image"])
            question = data["input"]

            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question}",
                    "images": [imgp],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            )

            input_embeds = model.prepare_inputs_embeds(**prepare_inputs).float()

            n_token = input_embeds.shape[1]

            attention_mask = np.ones((1, n_token), dtype=np.float32)
            position_ids = np.arange(n_token, dtype=np.int64).reshape(1, -1)
            num_logits_to_keep = np.array(n_token - 1).astype(np.int32).reshape(1)

            input_embeds = input_embeds.float()
            path_input_embeds = '{}/{}_{}.npy'.format(dataset_out_path_np, 'input_embeds', i)
            np.save(path_input_embeds, input_embeds.cpu().detach().numpy())

            path_attention_mask = '{}/{}_{}.npy'.format(dataset_out_path_np, 'attention_mask', i)
            np.save(path_attention_mask, attention_mask)

            path_position_ids = '{}/{}_{}.npy'.format(dataset_out_path_np, 'position_ids', i)
            np.save(path_position_ids, position_ids)

            path_num_logits_to_keep = '{}/{}_{}.npy'.format(dataset_out_path_np, 'num_logits_to_keep', i)
            np.save(path_num_logits_to_keep, num_logits_to_keep)

            f.write(os.path.abspath(path_input_embeds) + ' '
                + os.path.abspath(path_attention_mask) + ' '
                + os.path.abspath(path_position_ids) + ' '
                + os.path.abspath(path_num_logits_to_keep) + ' ')            
            f.write('\n')

    if grq_data:
        grq_data_path = '../../../../datasets/MMBench/llm/grq_inputs.json'
        gen_grq_input_embeds_dataset(datasets, dataset_out_path_np, grq_data_path)