# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import math
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling

from ...data_gd import get_dataset, split_dataset
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import CustomTrainer
from ...expert_setting.expert_setting import get_unlearned_expert_config, get_idx_deepseek, set_model_param

expert_layer_num = 26

        
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def run_gd(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    #print('data_args.dataset',data_args.dataset)
    #dataset = get_dataset(model_args, data_args, training_args, stage="pt", **tokenizer_module)
    # 0822 change
    dataset_list = [ds.strip() for ds in data_args.dataset.split(",")]

    print('+++++++++++++++++++++++++++++++++++++++++++dataset_list+++++++++++++++++++++++++++++++++++++++++++++++', dataset_list)
    data_args.dataset = dataset_list[0]
    forget_set = get_dataset(model_args, data_args, training_args, stage="pt", **tokenizer_module)
    
    data_args.dataset = dataset_list[1]
    retain_set = get_dataset(model_args, data_args, training_args, stage="pt", **tokenizer_module)
    
        
    '''
    forget_set=[]
    retain_set=[]
    print("workflow mixed dataset1 (forget):", dataset[0])
    print("workflow mixed dataset2 (retain):", dataset[1])
    for i in range(len(dataset)):
        if i % 2 ==0:
            
            forget_set.append(dataset[i])
        else:
            retain_set.append(dataset[i])
    '''
    
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    if finetuning_args.expert_training_mode != 0:
        finetuning_args = get_unlearned_expert_config(finetuning_args)
        model = set_model_param(model, finetuning_args)
    '''
    for name, param in model.named_parameters():

        layer_idx, expert_idx = get_idx_deepseek(name)
        if layer_idx == None:
            param.requires_grad = False
            continue
        if layer_idx > expert_layer_num:
            param.requires_grad = False
            continue
        if expert_idx in expert_config[str(layer_idx)]:
            param.requires_grad = True
            print('layer_idx, expert_idx', layer_idx, expert_idx)
        else:
            param.requires_grad = False'''
            
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # set grad of targeting layers
    '''for name, param in model.named_parameters():
        if finetuning_args.freeze_experts_layers == True:
        
            if 'mlp.gate.weight' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
              
        if finetuning_args.freeze_gating_layers == True:
            if 'mlp.gate.weight' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True'''

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **tokenizer_module,
        **split_dataset(forget_set, data_args, training_args),
        #forget_set=forget_set,
        retain_set=retain_set
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
