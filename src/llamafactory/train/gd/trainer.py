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

from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler


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
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], retain_set, **kwargs
    ) -> None:
        self.counter = 0
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.retain_set = retain_set

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
        retain_batch = {}
        
        #logger.info(inputs)
        for forget_sample in inputs:

            for key in inputs:
            #inputs[key][0] = torch.tensor(self.forget_set[forget_pos][key]).to(inputs[key][0].device)
            #inputs[key][1] = torch.tensor(self.retain_set[retain_pos][key]).to(inputs[key][1].device)
            
            #forget_batch[key] = inputs[key][::2]
            #retain_batch[key] = inputs[key][1::2]
                if key == 'labels':
                    retain_pos = self.counter % len(self.retain_set)
                    retain_batch[key] = torch.tensor(self.retain_set[retain_pos]['input_ids']).to(inputs[key][0].device)
                    self.counter += 1
                    
                else:
                    retain_pos = self.counter % len(self.retain_set)
                    retain_batch[key] = torch.tensor([self.retain_set[retain_pos][key]]).to(inputs[key][0].device)
                    self.counter += 1
            # print(forget_batch['input_ids'][0].to("cpu").tolist())
            '''
            if forget_batch['input_ids'][0].to("cpu").tolist() not in self.forget_set:
                logger.info("forget_batch is Not in forget_set !!!!!!!!!!")
                

            if forget_batch['input_ids'][0].to("cpu").tolist() in self.forget_set:
                logger.info('forget_batch is in forget_set')
                
            if retain_batch['input_ids'][0].to("cpu").tolist() not in self.retain_set:
                logger.info("retain_batch is Not in retain_set !!!!!!!!!!")
                print(retain_batch['input_ids'][0].to("cpu").tolist())

            if retain_batch['input_ids'][0].to("cpu").tolist() in self.retain_set:
                logger.info('retain_batch is in retain_set')'''

        
        if return_outputs:
            loss_forget, outputs_forget = super().compute_loss(model, inputs, return_outputs)
            loss_forget =-loss_forget
            loss_retain, outputs_retain = super().compute_loss(model, retain_batch, return_outputs)
        else:
            loss_forget = super().compute_loss(model, inputs, return_outputs)
            loss_forget =-loss_forget
            loss_retain = super().compute_loss(model, retain_batch, return_outputs)
        loss = loss_forget + loss_retain

        return (loss, outputs) if return_outputs else loss

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)
