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
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, \
                         MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .videollama2_arch import Videollama2MetaModel, Videollama2MetaForCausalLM


class Videollama2MistralConfig(MistralConfig):
    model_type = "videollama2_mistral"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "videollama2_mistral"


class Videollama2MistralModel(Videollama2MetaModel, MistralModel):
    config_class = Videollama2MistralConfig

    def __init__(self, config: MistralConfig):
        super(Videollama2MistralModel, self).__init__(config)


class Videollama2MistralForCausalLM(MistralForCausalLM, Videollama2MetaForCausalLM):
    config_class = Videollama2MistralConfig

    def __init__(self, config, **kwargs):
        super(MistralForCausalLM, self).__init__(config)
        self.model = Videollama2MistralModel(config) # 32层
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None: # <3> 非None跳过
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        outputs = super().forward( # <4>
            input_ids=input_ids, # None
            attention_mask=attention_mask, # torch.Size([1, 998])
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # torch.Size([1, 998, 4096])
            labels=labels, # None
            use_cache=use_cache, # True
            output_attentions=output_attentions, # False
            output_hidden_states=output_hidden_states, # 设置为True了
            return_dict=return_dict, # True
        )

        outputs.labels = labels
        self.extracted_hidden_states = outputs.hidden_states[-1][:, self.vtoken_start : self.vtoken_end + 1]
        return outputs

    @torch.no_grad() # 要对这个函数进行替换
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

        if images is not None:
            (
                input_ids,
                attention_mask, # torch.Size([1, 998])
                past_key_values,
                inputs_embeds, # torch.Size([1, 998, 4096])
                _,
                self.vtoken_start,
                self.vtoken_end,
                # visual_embedding,
                hidden_states
                # features_12, 
                # features_22,
                # features_15, 
                # features_27
                
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs, # torch.Size([1, 154])
                attention_mask=attention_mask, # torch.Size([1, 154])
                past_key_values=None,
                labels=None,
                images=images # list torch.Size([8, 3, 336, 336])
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate( # 调用GenerationMixin的generate方法 <1>
            position_ids=position_ids, # None
            attention_mask=attention_mask, # torch.Size([1, 998])
            inputs_embeds=inputs_embeds, # torch.Size([1, 998, 4096])
            output_hidden_states=True,
            **kwargs
        )

    def get_mm_features(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        use_cache = True,
        output_hidden_states: bool = True,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_select_layer = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                attention_mask, # torch.Size([1, 998])
                past_key_values,
                inputs_embeds, # torch.Size([1, 998, 4096])
                _,
                self.vtoken_start,
                self.vtoken_end,
                # visual_embedding,
                hidden_states
                # features_12, 
                # features_22,
                # features_15, 
                # features_27
                
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs, # torch.Size([1, 154])
                attention_mask=attention_mask, # torch.Size([1, 154])
                past_key_values=None,
                labels=None,
                images=images, # list torch.Size([8, 3, 336, 336])
                vision_select_layer=vision_select_layer
            )

        outputs = super().forward( # <4>
            input_ids=input_ids, # None
            attention_mask=attention_mask, # torch.Size([1, 998])
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # torch.Size([1, 998, 4096])
            labels=labels, # None
            use_cache=use_cache, # True
            output_attentions=output_attentions, # False
            output_hidden_states=output_hidden_states, # 设置为True了
            return_dict=return_dict, # True
        )

        outputs.labels = labels
        self.extracted_hidden_states = outputs.hidden_states[-1][:, self.vtoken_start : self.vtoken_end + 1]
        return self.extracted_hidden_states, hidden_states#features_12, features_22, features_15, features_27

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None) # <2> none
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("videollama2_mistral", Videollama2MistralConfig)
AutoModelForCausalLM.register(Videollama2MistralConfig, Videollama2MistralForCausalLM)
