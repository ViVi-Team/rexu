from types import MethodType
from typing import TYPE_CHECKING, Optional
from torch import nn
import torch
from transformers import Trainer
import torch.nn.functional as F
from copy import deepcopy
import os
from ...extras.logging import get_logger
from ..trainer_utils import align_scheduler_with_optimizer_param_groups, create_custom_optimzer, create_custom_scheduler
import deepspeed


if TYPE_CHECKING:
    import torch

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


def compute_npo_objective(
    forget_loss_current: torch.Tensor,
    forget_loss_reference: torch.Tensor,
    beta: float,
    retain_loss: torch.Tensor | None = None,
    retain_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return NPO+retain total loss and the unregularized NPO component."""
    neg_log_ratios = forget_loss_current - forget_loss_reference
    npo_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta
    total_loss = npo_loss
    if retain_loss is not None and retain_weight:
        total_loss = total_loss + float(retain_weight) * retain_loss
    return total_loss, npo_loss


class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(
        self,
        ref_model,
        finetuning_args: "FinetuningArguments",
        retain_set=None,
        retain_collator=None,
        retain_weight: float = 0.0,
        pairwise_forget: bool = True,
        **kwargs,
    ) -> None:
        self.retain_set = retain_set
        self.retain_collator = retain_collator
        self.retain_weight = float(retain_weight)
        self.pairwise_forget = bool(pairwise_forget)
        self.retain_counter = 0
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.ref_model = ref_model
        self.beta = finetuning_args.pref_beta
        self._printed_npo_guard = False

        if self.beta <= 0:
            raise ValueError("NPO requires pref_beta > 0.")
        
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    if os.environ.get("NPO_REF_PLAIN_GPU", "1").lower() in {"1", "true", "yes"}:
                        self.ref_model = self._prepare_plain_reference_model(self.ref_model)
                    else:
                        self.ref_model = self._prepare_deepspeed(self.ref_model)
                    self.ref_model.eval()
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            raise ValueError("NPO requires a frozen reference model. Set ref_model or use full/expert finetuning.")

        if self.retain_weight > 0 and (self.retain_set is None or self.retain_collator is None):
            raise ValueError("NPO retain_weight > 0 requires retain_set and retain_collator.")
        logger.info(
            "Using official NPO loss with frozen reference model, beta=%.6g, retain_weight=%.6g, retain_examples=%s.",
            self.beta,
            self.retain_weight,
            len(self.retain_set) if self.retain_set is not None else 0,
        )
                
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)
            
    def _prepare_plain_reference_model(self, model):
        """Keep the frozen NPO reference outside DeepSpeed.

        DeepSpeed ZeRO-3 can leave DeepSeek MoE expert params in-flight on the
        no-grad reference forward. A plain bf16 reference copy fits on a B200
        and avoids ZeRO's parameter coordinator entirely.
        """
        logger.info("Preparing NPO reference model without DeepSpeed on device %s.", self.args.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.to(self.args.device)
        return model

    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        zero_config = config_kwargs.get("zero_optimization", {})

        if os.environ.get("NPO_REF_DEEPSPEED_NO_OFFLOAD", "1").lower() not in {"0", "false", "no"}:
            # The frozen reference model is used only for no-grad forward passes.
            # CPU parameter offload can leave MoE expert params inflight in
            # DeepSpeed's reference engine, so keep the ref weights resident while
            # allowing the trainable model to use the configured offload policy.
            if isinstance(zero_config, dict):
                zero_config.pop("offload_param", None)
                zero_config.pop("offload_optimizer", None)
            config_kwargs.pop("zero_optimization.offload_param", None)
            config_kwargs.pop("zero_optimization.offload_optimizer", None)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return align_scheduler_with_optimizer_param_groups(super().create_scheduler(num_training_steps, optimizer))

    def get_batch_loss(self, output, labels):
        shifted_labels = labels[..., 1:].contiguous()
        output = output[..., :-1, :].contiguous()

        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        # get the sum loss for each sequence in a batch
        loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

        return loss

    def _forget_batch_from_inputs(self, inputs):
        """Use the rejected half of LLaMAFactory pairwise batches as negative/forget samples."""
        if not self.pairwise_forget:
            return inputs
        batch_size = inputs["input_ids"].shape[0]
        if batch_size % 2 != 0:
            return inputs

        half = batch_size // 2
        forget_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.shape[0] == batch_size:
                forget_inputs[key] = value[half:]
            else:
                forget_inputs[key] = value
        return forget_inputs

    def _forward_model(self, model, batch):
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            labels=batch["labels"],
        )

    def _next_retain_batch(self, batch_size: int, device: torch.device):
        if self.retain_set is None or self.retain_collator is None or self.retain_weight <= 0:
            return None
        examples = [
            self.retain_set[(self.retain_counter + offset) % len(self.retain_set)]
            for offset in range(batch_size)
        ]
        self.retain_counter = (self.retain_counter + batch_size) % len(self.retain_set)
        batch = self.retain_collator(examples)
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def compute_loss(self, model, inputs, return_outputs=False):
        # Official NPO implementation:
        #   neg_log_ratios = NLL_theta(y|x) - NLL_ref(y|x)
        #   L_NPO = -(2/beta) * mean(logsigmoid(beta * neg_log_ratios))
        # See licong-lin/negative-preference-optimization, TOFU/dataloader.py.
        forget_inputs = self._forget_batch_from_inputs(inputs)
        if not self._printed_npo_guard:
            logger.info(
                "NPO guard: batch_size=%s, forget_batch_size=%s, pairwise_rejected_half=%s, ref_model_frozen=True",
                inputs["input_ids"].shape[0],
                forget_inputs["input_ids"].shape[0],
                self.pairwise_forget and inputs["input_ids"].shape[0] != forget_inputs["input_ids"].shape[0],
            )
            self._printed_npo_guard = True

        outputs = self._forward_model(model, forget_inputs)
        forget_loss_current = self.get_batch_loss(outputs.logits, forget_inputs["labels"])

        with torch.no_grad():
            forget_outputs_oracle = self._forward_model(self.ref_model, forget_inputs)
            forget_logits_oracle = forget_outputs_oracle.logits
            forget_loss_oracle = self.get_batch_loss(forget_logits_oracle, forget_inputs["labels"])
        
        retain_batch = self._next_retain_batch(
            forget_inputs["input_ids"].shape[0],
            forget_inputs["input_ids"].device,
        )
        retain_loss = None
        if retain_batch is not None:
            retain_outputs = self._forward_model(model, retain_batch)
            retain_loss = retain_outputs.loss

        loss, npo_loss = compute_npo_objective(
            forget_loss_current,
            forget_loss_oracle,
            self.beta,
            retain_loss,
            self.retain_weight,
        )
        self._last_npo_components = {
            "npo_loss": float(npo_loss.detach().cpu()),
            "retain_loss": float(retain_loss.detach().cpu()) if retain_loss is not None else 0.0,
            "retain_weight": self.retain_weight,
            "total_loss": float(loss.detach().cpu()),
        }

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float]) -> None:
        if hasattr(self, "_last_npo_components"):
            logs = {**logs, **{f"npo/{key}": value for key, value in self._last_npo_components.items()}}
        super().log(logs)
