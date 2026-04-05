# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .videollama2_arch import Videollama2MetaModel, Videollama2MetaForCausalLM


class Videollama2Qwen2Config(Qwen2Config):
    model_type = "videollama2_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "videollama2_qwen2"


class Videollama2Qwen2Model(Videollama2MetaModel, Qwen2Model):
    config_class = Videollama2Qwen2Config

    def __init__(self, config: Videollama2Qwen2Config):
        super(Videollama2Qwen2Model, self).__init__(config)


class Videollama2Qwen2ForCausalLM(Qwen2ForCausalLM, Videollama2MetaForCausalLM):
    config_class = Videollama2Qwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = Videollama2Qwen2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 3584->152064

        self.vtoken_start = None  
        self.vtoken_end = None
        self.extracted_hidden_states = None 
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None, # 用于加速生成过程的缓存信息（避免重新计算历史 token 的信息）
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None: # torch.Size([1, 1391, 3584]) -> None
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                _,
                _,
                _
                
            ) = self.prepare_inputs_labels_for_multimodal( # ⑤⑨
                input_ids, # -> torch.Size([1, 1])这一轮没有进行实际操作->torch.Size([1, 1])，一直是最新生成的
                attention_mask, # -> torch.Size([1, 1392])->[1, 1393]
                past_key_values,
                labels, # ->None
                images # ->None
            )

        output = super().forward( # ②⑥⑩
            input_ids=input_ids, # None -> tensor([[785]], device='cuda:0')->tensor([[2766]], device='cuda:0')
            attention_mask=attention_mask, # torch.Size([1, 1390])->torch.Size([1, 1391]) 标记token全部有效->torch.Size([1, 1393])
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # torch.Size([1, 1390, 3584]) -> None
            labels=labels, # None
            use_cache=use_cache, # True
            output_attentions=output_attentions, # False
            output_hidden_states=output_hidden_states, # False
            return_dict=return_dict, # True
        )
        
        # self.extracted_hidden_states = output.hidden_states[-1][:, self.vtoken_start : self.vtoken_end + 1]
        # output['extracted_hidden_states'] = hidden_states
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None: # 第一次进入函数
            (
                input_ids, # None
                attention_mask, # torch.Size([1, 1390])
                past_key_values, # None
                inputs_embeds, # torch.Size([1, 1390, 3584])
                _,
                self.vtoken_start,
                self.vtoken_end,
                # visual_embedding,
                hidden_states
                # features_12, 
                # features_22,
                # features_15, 
                # features_27
                
            ) = self.prepare_inputs_labels_for_multimodal( # 视觉编码，整合文本图像token到一个序列里
                input_ids=inputs, # torch.Size([1, 39])
                attention_mask=attention_mask, # torch.Size([1, 39])
                past_key_values=None, # None
                labels=None,
                images=images # torch.Size([1, 3, 384, 384])
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate( # 调用GenerationMixin的generate方法
            position_ids=position_ids, # None
            attention_mask=attention_mask, # [1,1390]
            inputs_embeds=inputs_embeds, # [1,1390,3584]
            output_hidden_states=True,
            **kwargs
        ) # !!! 使用自回归的方式调用 forward，逐步生成新 token 直到达到指定条件（如生成结束标志或达到最大生成长度）。generate 函数利用 forward 函数来计算每一步的输出概率，决定下一个生成的 token。

    # @torch.no_grad()
    def get_mm_features(
        self,
        inputs: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        images: torch.FloatTensor = None,
        use_cache=True,
        output_hidden_states: bool = True,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_select_layer = -1,
        vision_select_layer = None,
        **kwargs
    ) -> torch.Tensor:
        
        if images is not None:
            (
                input_ids, # None
                attention_mask, # torch.Size([1, 1390])
                past_key_values, # None
                inputs_embeds, # torch.Size([1, 1390, 3584])
                _,
                self.vtoken_start,
                self.vtoken_end,
                # visual_embedding,
                hidden_states
                # features_12, 
                # features_22,
                # features_15, 
                # features_27
                
            ) = self.prepare_inputs_labels_for_multimodal( # 视觉编码，整合文本图像token到一个序列里
                input_ids=inputs, # torch.Size([1, 39])
                attention_mask=attention_mask, # torch.Size([1, 39])
                past_key_values=None, # None
                labels=None,
                images=images, # torch.Size([1, 3, 384, 384]),
                vision_select_layer=vision_select_layer
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            
        output = super().forward( 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            # past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, 
            labels=labels, # None
            use_cache=use_cache, # True
            output_attentions=output_attentions, # False
            output_hidden_states=output_hidden_states, # False
            return_dict=return_dict, # True
        )
        
        if output_hidden_states:
            self.extracted_hidden_states = output.hidden_states[hidden_select_layer][:, self.vtoken_start : self.vtoken_end + 1]
        # output['extracted_hidden_states'] = hidden_states
        return self.extracted_hidden_states, hidden_states #features_12, features_22, features_15, features_27
        
        
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs): # 用于准备生成过程中每一步的输入数据，以便传递给模型的 forward 方法。这个方法在自回归生成过程中被 generate 方法多次调用，用来更新模型的输入，包括文本序列、注意力掩码和缓存的 past_key_values。 # 在自回归生成过程中，每次调用 forward 时会计算当前输入的注意力和前馈网络层的输出。past_key_values 缓存了前一步生成过程中所有层的 key 和 value，因此在后续生成中可以直接使用它们，无需重复计算。 input_ids：torch.Size([1, 0])，inputs_embeds：torch.Size([1, 1391, 3584])
        images = kwargs.pop("images", None) # None ①④⑧
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs # inputs_embeds一直是完整的多模态输入，input_ids从[]开始增加 
        ) # 父类方法会根据 input_ids 和 past_key_values 生成 attention_mask、position_ids 等输入，并构建一个 _inputs 字典，该字典包含生成任务所需的所有输入。 第一次和第二次调用时生成的字典键不同，第一次调用 prepare_inputs_for_generation 是为了初始化生成过程。在这个阶段，模型可能会直接处理初始的 input_ids 或 inputs_embeds，通常还不需要缓存 past_key_values。然而在后续的生成过程中，模型会逐步生成每个新 token，每次生成一个 token 后再次调用 prepare_inputs_for_generation，此时 past_key_values 会起作用，因为它包含了之前生成步骤的注意力信息缓存，以加速生成。第一次调用：通常没有 past_key_values（即为 None），此时 prepare_inputs_for_generation 会根据完整的 input_ids 生成 attention_mask 和 position_ids，并将这些信息保存在生成的字典中。第二次及后续调用：past_key_values 不再为空，模型使用 past_key_values 加速生成。此时，input_ids 可能仅包含最新生成的 token，因为历史 token 的信息已存储在 past_key_values 中。此时生成的字典可能会省略 attention_mask 等键，直接使用缓存的值。如果直接提供 inputs_embeds，则通常不会需要 input_ids。在多模态任务中，例如 Videollama2Qwen2ForCausalLM，可能会直接传入视觉特征的嵌入作为 inputs_embeds，而不再使用 input_ids。这种情况导致在第一次调用和第二次调用时生成的字典键可能不同，因为 inputs_embeds 已包含了嵌入信息，不需要再从 input_ids 生成。
        if images is not None:
            _inputs['images'] = images
        # if 'vtoken_start' in kwargs:
        #     _inputs['vtoken_start'] = kwargs['vtoken_start']
        # if 'vtoken_end' in kwargs:
        #     _inputs['vtoken_end'] = kwargs['vtoken_end']
        return _inputs # 'inputs_embeds':[1,1390,3584]; 'position_ids':[1,1390]; 'attention_mask':[1,1390]; 'cache_position':[1390]; 'past_key_values'; 'use_cache' # 此后，input_ids仅包含最新生成，position_ids为最新位置，attention_mask在初始尺寸上逐渐增加，


AutoConfig.register("videollama2_qwen2", Videollama2Qwen2Config)
AutoModelForCausalLM.register(Videollama2Qwen2Config, Videollama2Qwen2ForCausalLM)
