import torch
import torch.nn.functional as F
import time
import sys, os, copy
from numpy import nonzero
import numpy as np

from utils import AverageMeter
from videollama2.constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP
from videollama2.mm_utils import tokenizer_multimodal_token


def val_epoch(epoch, nEpochs, data_loader, model, decoder, tokenizer, optimizer, criterion, opt, val_loss_min, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()
    decoder.eval()

    with torch.no_grad():

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        nll_losses = AverageMeter()
        CC_losses = AverageMeter()
        KL_losses = AverageMeter()
        BCE_losses = AverageMeter()

        end_time = time.time()

        batch = 0

        for i, (data, target) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            target['salmap'] = target['salmap'].cuda()
            if opt.precision == 'f16':
                target['salmap'] = target['salmap'].half() # torch.Size([1, 208, 208])
            # elif opt.precision == 'bf16':
            #     target['salmap'] = target['salmap'].to(torch.bfloat16) # torch.Size([1, 208, 208])
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
        
            loss, nll_loss,  CC_loss, KL_loss, BCE_loss = criterion(output, target['salmap'])
            losses.update(loss)
            nll_losses.update(nll_loss)
            CC_losses.update(CC_loss)
            KL_losses.update(KL_loss)
            BCE_losses.update(BCE_loss)
            

            if (i+1) % opt.update_step == 0:
                batch = batch + 1
                
                batch_time.update(time.time() - end_time)
                end_time = time.time()
                
                print('Val: Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                'nll {nll_losses.val:.4f} ({nll_losses.avg:.4f})\t'
                'CC {CC_losses.val:.4f} ({CC_losses.avg:.4f})\t'
                'KL {KL_losses.val:.4f} ({KL_losses.avg:.4f})\t'
                'BCE {BCE_losses.val:.4f} ({BCE_losses.avg:.4f})\t'.format(epoch, i+1, len(data_loader),
                    batch_time=batch_time,
                    losses=losses,
                    nll_losses=nll_losses,
                    CC_losses=CC_losses,
                    KL_losses=KL_losses,
                    BCE_losses=BCE_losses
                    ))


        logger.log({
            'epoch': epoch,
            'loss': "{:.4f}".format(losses.avg),
            'nll': "{:.4f}".format(nll_losses.avg),
            'CC': "{:.4f}".format(CC_losses.avg),
            'KL': "{:.4f}".format(KL_losses.avg),
            'BCE': "{:.4f}".format(BCE_losses.avg),
            'lr_p':"{}".format(optimizer.state_dict()['param_groups'][0]['lr']),
            'lr_d':"{}".format(optimizer.state_dict()['param_groups'][1]['lr'])
        })

        val_loss = losses.avg

        if val_loss <= val_loss_min or CC_losses.avg < -0.71:
            save_file_path = os.path.join(opt.result_path, opt.model_name,
                                        '{}_{}.pth'.format(opt.model_name, epoch))
            if not opt.mm_train:
                states = {
                    'epoch': epoch + 1,
                    'readout_state_dict': model.model.mm_projector.readout.state_dict(),
                    'decoder_state_dict': decoder.state_dict()
                }
            else:
                states = {
                    'epoch': epoch + 1,
                    'mm_projector_state_dict': model.model.mm_projector.state_dict(),
                    'decoder_state_dict': decoder.state_dict()
                }
            torch.save(states, save_file_path)
            val_loss_min = val_loss

        return losses.avg, val_loss_min
