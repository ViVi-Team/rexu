# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
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

import copy
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling

from ...data import PairwiseDataCollatorWithPadding, get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomTrainer
from ...expert_setting.expert_setting import get_unlearned_expert_config, get_idx_deepseek, set_model_param

def get_idx_deepseek(layer_name):
    name_list=layer_name.split('.')
    if len(name_list)<4:
        return None, None

    if name_list[1]=='layers' and name_list[4]=='experts':
        return int(name_list[2]), int(name_list[5])
    else:
        return None, None
        
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


def run_npo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset_names = [name.strip() for name in str(data_args.dataset).split(",") if name.strip()]
    forget_data_args = copy.deepcopy(data_args)
    forget_data_args.dataset = dataset_names[0]
    pairwise_forget = not forget_data_args.dataset.startswith("rwku_forget_")
    forget_stage = "rm" if pairwise_forget else "pt"
    dataset = get_dataset(model_args, forget_data_args, training_args, stage=forget_stage, **tokenizer_module)

    retain_name = dataset_names[1] if len(dataset_names) > 1 else None
    retain_weight = float(finetuning_args.pref_ftx)
    if forget_data_args.dataset.startswith("wmdp_cyber"):
        # SEUF Appendix A specifies lambda=1 for NPO. Historical generated
        # configs omitted pref_ftx, so preserve paper behavior for queued WMDP
        # jobs while leaving non-WMDP NPO configurations unchanged.
        retain_name = retain_name or "wmdp_cyber_retain"
        retain_weight = retain_weight or 1.0
    retain_set = None
    retain_collator = None
    if retain_name is not None and retain_weight > 0:
        retain_data_args = copy.deepcopy(data_args)
        retain_data_args.dataset = retain_name
        retain_set = get_dataset(model_args, retain_data_args, training_args, stage="pt", **tokenizer_module)
        retain_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print(
        f"[NPO] forget_dataset={forget_data_args.dataset} retain_dataset={retain_name} "
        f"beta={finetuning_args.pref_beta} retain_weight={retain_weight} "
        f"forget_batch_format={'pairwise_rejected' if pairwise_forget else 'causal_lm'}",
        flush=True,
    )
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    print("+================================loading topk training====================================")
    counter=0
    

    if pairwise_forget:
        data_collator = PairwiseDataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    finetuning_args.use_ref_model = True
    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
            #ref_model = load_model(tokenizer, model_args, finetuning_args, training_args.do_eval)
    else:
        ref_model = None
        
    if ref_model is None:
        ref_model = copy.deepcopy(model)
    if finetuning_args.expert_training_mode != 0:
        finetuning_args = get_unlearned_expert_config(finetuning_args)
        model = set_model_param(model, finetuning_args)

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        retain_set=retain_set,
        retain_collator=retain_collator,
        retain_weight=retain_weight,
        pairwise_forget=pairwise_forget,
        **split_dataset(dataset, forget_data_args, training_args),
    )

    # Training
    if training_args.do_train:
        print("start training")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
