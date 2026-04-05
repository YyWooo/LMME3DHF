import torch
import torch.nn.functional as F
import time
import os
import sys
import copy
import numpy as np
from imageio import imwrite
from videollama2.mm_utils import save_salmap
from PIL import Image
from videollama2.mm_utils import tokenizer_multimodal_token
from videollama2.constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP


from utils import AverageMeter




def test(data_loader, model, decoder, tokenizer, opt):
    print('Test')

    decoder.eval()
    filename = os.path.basename(opt.resume_path)
    base_name = os.path.splitext(filename)[0]
    filefolder = os.path.join(opt.result_path, opt.model_name, base_name, opt.dataset, opt.phase)
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)

    with torch.no_grad():

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end_time = time.time()
        
        time_record1 = time.time()
        for i, (data, target) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            # target['salmap'] = target['salmap'].cuda().float()
            instruct = data['instruct'][0]
            modal = data['modal'][0]
            
            # 1. text preprocess (tag process & generate prompt).
            if modal == 'image':
                modal_token = DEFAULT_IMAGE_TOKEN
            elif modal == 'video':
                modal_token = DEFAULT_VIDEO_TOKEN
            elif modal == 'text':
                modal_token = ''
            else:
                raise ValueError(f"Unsupported modal: {modal}")

            # 1. vision preprocess (load & transform image or video).
            if modal == 'text':
                tensor = None
            else:
                if opt.precision in ['f16', 'f32']:
                    tensor = data['vision'].half().cuda()
                elif opt.precision in ['bf16', 'ori']:
                    tensor = data['vision'].to(torch.bfloat16).cuda()#  # 注意一下data的处理
                tensor = tensor.squeeze(0) # torch.Size([1, 3, 384, 384])
                tensor = [(tensor, modal)]

            # 2. text preprocess (tag process & generate prompt).
            if isinstance(instruct, str):
                message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
            elif isinstance(instruct, list):
                message = copy.deepcopy(instruct)
                message[0]['content'] = modal_token + '\n' + message[0]['content']
            else:
                raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

            if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
                system_message = [
                    {'role': 'system', 'content': (
                    """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
                    """\n"""
                    """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
                    }
                ]
            else:
                system_message = []

            message = system_message + message
            prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) # '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhat is the woman wearing, what is she doing, and how does the image feel?<|im_end|>\n<|im_start|>assistant\n'

            input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda() # [b, l]
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda() # attention_masks 会标记哪些 token 是有效的，哪些是填充 token，从而帮助模型在计算注意力（attention）时专注于实际的内容。 [b, l]

            if opt.ver in ["ver4", "ver5", "ver6", "ver7", "ver8", "ver10", "ver11", "ver12", "ver13", "ver14", "ver16", "ver17", "ver18", "ver19"]:
                vision_select_layer = [-2,-12,-22]
            elif opt.ver in ["ver9"]:
                vision_select_layer = [-2,-15,-27]
            elif opt.ver in ["ver21"]:
                vision_select_layer = [-1,-2,-12,-22]
            elif opt.ver in ["ver15"]:
                vision_select_layer = [-2,-10,-18,-26]
                
            if opt.ver in ["ver21"]:
                output_hidden_states=False
            else:
                output_hidden_states=True

            ## 替换成自己的特征提取函数
            with torch.inference_mode():
                mm_features, hidden_states = model.get_mm_features( # 调用Videollama2Qwen2ForCausalLM的generate方法
                    input_ids, # torch.Size([1, 39])
                    attention_mask=attention_masks, # torch.Size([1, 39])
                    images=tensor, # torch.Size([1, 3, 384, 384])
                    use_cache=True,
                    output_hidden_states=output_hidden_states,
                    hidden_select_layer=opt.hidden_select_layer,
                    vision_select_layer=vision_select_layer
                )
                
            if opt.ver in ["ver21"]:
                hidden_states = [tensor.float() for tensor in hidden_states]
            elif opt.precision in ['f32', 'bf16', 'ori']:
                mm_features = mm_features.float()
                hidden_states = [tensor.float() for tensor in hidden_states]

            device = next(decoder.parameters()).device
            mm_features = mm_features.to(device)
            hidden_states = [h.to(device) for h in hidden_states]

            if opt.ver == "ver2":
                output = decoder(mm_features)
            elif opt.ver == "ver21":
                output = decoder(hidden_states)
            else:
                output = decoder(mm_features, hidden_states)
                
            if modal == 'image':
                save_salmap(output, os.path.join(filefolder, target['name'][0]))
            elif modal == 'video':
                if not os.path.exists(os.path.join(filefolder, target['name'][0])):
                    os.mkdir(os.path.join(filefolder, target['name'][0]))
                save_salmap(output, os.path.join(filefolder, target['name'][0], '{:03d}.png'.format(target['index'][0])))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))
        time_infer = time.time() - time_record1
        print("*****************   infer-time =",time_infer,"sec   **************")

