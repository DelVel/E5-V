import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from einops import rearrange, reduce
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
)
from transformers.debug_utils import DebugOption
from transformers.integrations.deepspeed import deepspeed_init
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import (
    EvalLoopOutput,
    denumpify_detensorize,
    speed_metrics,
)

from src.data import (
    custom_collate_fn,
    recall_at_k,
)


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensors
    return torch.cat(GatherLayer.apply(tensors))


class SentembTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        embs = inputs.pop("label")
        outputs = model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
            drop=True,
        ).hidden_states[-1][:, -1, :]
        embs = torch.stack(embs)

        if dist.is_initialized():
            outputs = all_gather_with_grad(outputs.contiguous())
            embs = all_gather_with_grad(embs.contiguous())

        outputs = F.cosine_similarity(
            outputs.unsqueeze(1), outputs.unsqueeze(0), dim=-1
        )
        embs = F.cosine_similarity(embs.unsqueeze(1), embs.unsqueeze(0), dim=-1)

        # normalize clip_simmat to fit in [-1, 1]
        clip_mat_max = embs.amax()
        clip_mat_min = embs.amin()
        embs = 2 * (embs - clip_mat_min) / (clip_mat_max - clip_mat_min) - 1

        loss = ((embs - outputs) ** 2).mean()

        return (loss,) if return_outputs else loss

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, data.Dataset]] = None
    ) -> data.DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        # Change to use another collate fn
        # data_collator = self.data_collator
        data_collator = lambda x: custom_collate_fn(x, self.processing_class)

        if transformers.utils.is_datasets_available() and isinstance(
            eval_dataset, datasets.Dataset
        ):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = data.DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        data = inputs[0]
        with torch.no_grad():
            with self.compute_loss_context_manager():
                emb = model(
                    **data, output_hidden_states=True, return_dict=True
                ).hidden_states[-1][:, -1, :]

        return emb, inputs[1]

    def evaluate(
        self,
        eval_dataset: Optional[Union[data.Dataset, Dict[str, data.Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self.processing_class.drop = False
        metric_key_prefix = "eval_fiq_dress"
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset_q = self.eval_dataset["fiq_dress_query"]
        eval_dataset_t = self.eval_dataset["fiq_dress_target"]

        eval_dataloader_q = self.get_eval_dataloader(eval_dataset_q)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader_q = tpu_spmd_dataloader(eval_dataloader_q)

        eval_dataloader_t = self.get_eval_dataloader(eval_dataset_t)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader_t = tpu_spmd_dataloader(eval_dataloader_t)

        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            (eval_dataloader_q, eval_dataloader_t),
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            raise ImportError
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        self.processing_class.drop = True
        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval_fiq_dress",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        # logger.info(f"\n***** Running {description} *****")
        # if has_length(dataloader):
        #     logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        # else:
        #     logger.info("  Num examples: Unknown")
        # logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_q_embs = []
        all_q_ids = []

        tstep = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader[0]):
            tstep += 1
            emb, id_ = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            all_q_embs.extend(self.accelerator.gather_for_metrics(emb))
            all_q_ids.extend(self.accelerator.gather_for_metrics(id_))

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

        all_t_embs = []
        all_t_ids = []
        for step, inputs in enumerate(dataloader[1]):
            tstep += 1
            emb, id_ = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            all_t_embs.extend(self.accelerator.gather_for_metrics(emb))
            all_t_ids.extend(self.accelerator.gather_for_metrics(id_))

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

        all_q_embs = torch.stack(all_q_embs)
        all_q_ids = np.stack(all_q_ids)
        all_q_embs = reduce(all_q_embs, "(b 2) d -> b d", "sum")
        all_q_ids = rearrange(all_q_ids, "(b e) -> e b", e=2)[0]

        all_t_embs = torch.stack(all_t_embs)
        all_t_ids = np.stack(all_t_ids)

        all_q_embs = F.normalize(all_q_embs, dim=-1)
        all_t_embs = F.normalize(all_t_embs, dim=-1)
        scores = all_q_embs @ all_t_embs.t()

        positive_pairs = torch.from_numpy(all_q_ids[:, None] == all_t_ids[None, :]).to(
            self.accelerator.device, non_blocking=True
        )
        r10 = recall_at_k(scores, positive_pairs, 10).mean().item()
        r50 = recall_at_k(scores, positive_pairs, 50).mean().item()

        metrics = {"r10": r10, "r50": r50}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = (
                self.jit_compilation_time
            )
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = (
                self.model_preparation_time
            )

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=tstep,
        )
