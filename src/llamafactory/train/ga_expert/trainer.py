# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType
from typing import TYPE_CHECKING, Optional
import torch
from transformers import Trainer
import copy
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import align_scheduler_with_optimizer_param_groups, create_custom_optimzer, create_custom_scheduler
import os

if TYPE_CHECKING:
    import torch
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)

class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.original_gate=None

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        cache = []
        
        def hook(module, input, output):

            hidden_states=input[0]
            #print('hidden_states shape:',hidden_states.shape)
            import torch.nn.functional as F
            if str(self.finetuning_args.expert_training_mode)[0]== '9':
                bsz, seq_len, h = hidden_states.shape
                ### compute gating score
                hidden_states = hidden_states.view(-1, h)
                
            
                logits = F.linear(
                    hidden_states.type(torch.float32), module.weight.type(torch.float32), None
                )
                scores = logits.softmax(dim=-1, dtype=torch.float32)
                cache.append(scores)
            else: #Qwen
                #print('qwen output shape:', output.shape)
                logits = F.linear(
                    hidden_states.type(torch.float), module.weight.type(torch.float), None
                )
                scores = logits.softmax(dim=-1, dtype=torch.float)
                #scores = F.softmax(output, dim=1, dtype=torch.float) 
                
                cache.append(scores)
            expert=[]
            for key, var in self.finetuning_args.expert_config.items():
                expert.extend(var)
            
            chosen_experts = torch.zeros_like(cache[0])
            chosen_experts[:, expert] = 1.0

            self.chosen_loss = torch.nn.functional.mse_loss(
                    chosen_experts, scores
                    )

            return None 
    
        hook_handle = model.model.layers[self.finetuning_args.gating_layer].mlp.gate.register_forward_hook(hook)

        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs)
        loss = -loss
        
        hook_handle.remove()
        
        print('ga_expert_loss:', loss.item())
        loss += self.finetuning_args.chosen_gating_loss*self.chosen_loss
        print('chosen_gating_loss={}*{}'.format(self.finetuning_args.chosen_gating_loss, self.chosen_loss))
            
        return (loss, outputs) if return_outputs else loss

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return align_scheduler_with_optimizer_param_groups(super().create_scheduler(num_training_steps, optimizer))
