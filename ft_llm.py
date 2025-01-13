import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import accelerate
import datasets
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from einops import rearrange, reduce
from jsonargparse import CLI
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from torch.distributed.elastic.multiprocessing import errors
from torch.utils import data
from torch.utils.data import DataLoader
from transformers import (
    BitsAndBytesConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    set_seed,
)
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption
from transformers.integrations.deepspeed import deepspeed_init
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import (
    EvalLoopOutput,
    denumpify_detensorize,
    set_seed,
    speed_metrics,
)

from data import (
    custom_collate_fn,
    get_fiq_image_dataset,
    get_fiq_text_dataset,
    prompt_image_text,
    prompt_text,
    recall_at_k,
)


class LlavaCustom(LlavaForConditionalGeneration):
    """
    A custom model that allows both image and text inputs to be processed
    """

    def forward(self, *args, **kwargs):
        pixel_values = kwargs.get("pixel_values", None)
        if pixel_values is not None:
            return super().forward(*args, **kwargs)
        return self.language_model.forward(*args, **kwargs)


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


class DataCollator:
    def __init__(self, processor):
        self._processor: LlavaProcessor = processor

    def __call__(self, data_):
        text = [x["txt"] for x in data_]
        text = self._processor.batch_decode(
            self._processor(
                text=text,
                truncation=True,
                max_length=32,
                add_special_tokens=False,
            )["input_ids"]
        )

        text_templated = self._processor.apply_chat_template(
            [prompt_text(f"{x}\nSummary above sentence in one word:") for x in text],
            add_generation_prompt=True,
        )

        images = [x["jpg"] for x in data_]

        images_templated = self._processor.apply_chat_template(
            [prompt_image_text("Summary above image in one word:") for _ in images],
            add_generation_prompt=True,
        )
        images_processed = self._processor(
            images=images,
            text=images_templated + text_templated,
            pad_to_multiple_of=8,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        return images_processed


class SentembTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(
            **inputs, return_dict=True, output_hidden_states=True, use_cache=False
        ).hidden_states[-1][:, -1, :]
        outputs = rearrange(outputs, "(m b) d -> m b d", m=2)
        img_outputs, txt_outputs = outputs[0], outputs[1]

        if dist.is_initialized():
            img_outputs = all_gather_with_grad(img_outputs.contiguous())
            txt_outputs = all_gather_with_grad(txt_outputs.contiguous())

        query = img_outputs.unsqueeze_(1)
        target = txt_outputs.unsqueeze_(0)
        cos_sim = F.cosine_similarity(query, target, dim=-1) / 0.05

        labels = torch.arange(cos_sim.size(0), dtype=torch.long, device=cos_sim.device)

        loss = (
            F.cross_entropy(cos_sim, labels) + F.cross_entropy(cos_sim.t(), labels)
        ) / 2

        return (loss, txt_outputs, img_outputs) if return_outputs else loss

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


@dataclass
class LoraParams:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


def get_processor(model_name):
    processor = LlavaProcessor.from_pretrained(model_name)
    processor.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|>\n'}}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] + '<|end|>\n' }}{% endfor %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
    model_cfg = LlavaConfig.from_pretrained(model_name)
    processor.patch_size = model_cfg.vision_config.patch_size
    processor.vision_feature_select_strategy = model_cfg.vision_feature_select_strategy
    return processor


def get_model(
    model_name,
    lora_params: LoraParams,
    model_dtype,
):
    model = LlavaCustom.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        quantization_config=BitsAndBytesConfig(
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_quant_storage=model_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            load_in_4bit=True,
        ),
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_params.r,
            lora_alpha=lora_params.alpha,
            lora_dropout=lora_params.dropout,
            target_modules=lora_params.target_modules,
            exclude_modules="^(?!language_model).*$",
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return model


def get_dataset():
    cc3m = load_dataset("pixparse/cc3m-wds", split="train")
    cc3m = cc3m.shuffle()
    cc3m = cc3m.remove_columns(["__key__", "__url__"])
    return cc3m


@errors.record
def main(
    output_dir: str,
    run_name: str,
    lora: LoraParams,
    # training hyperparams
    per_device_train_batch_size: int = 64,
    gradient_accumulation_steps: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
    bf16: bool = True,
    # trainer parameters
    resume_from_checkpoint: bool = False,
):
    accelerator = accelerate.Accelerator()
    try:
        set_seed(42)

        if not accelerator.is_main_process:
            transformers.utils.logging.disable_progress_bar()
            datasets.disable_progress_bars()
        else:
            print("Progress bars are disabled in non-main processes.")

        output_dir: Path = Path(output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_dir)
        args = transformers.TrainingArguments(
            bf16=bf16,
            dataloader_num_workers=4,
            ddp_find_unused_parameters=False,
            deepspeed="ds_config.json",
            eval_strategy="steps",
            report_to="wandb",
            eval_steps=20,
            fp16=not bf16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            learning_rate=learning_rate,
            logging_steps=1,
            num_train_epochs=num_epochs,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            remove_unused_columns=False,
            run_name=run_name,
            save_steps=100,
            save_strategy="steps",
            save_total_limit=3,
            warmup_steps=100,
        )

        model_name = "xtuner/llava-phi-3-mini-hf"
        processor = get_processor(model_name)
        model = get_model(
            model_name,
            lora,
            torch.bfloat16 if bf16 else torch.float16,
        )
        if accelerator.is_main_process:
            model.print_trainable_parameters()

        train_data = get_dataset()
        data_collator = DataCollator(processor)
        trainer = SentembTrainer(
            args=args,
            data_collator=data_collator,
            model=model,
            processing_class=processor,
            train_dataset=train_data,
            eval_dataset={
                "fiq_dress_query": get_fiq_text_dataset(processor, "dress"),
                "fiq_dress_target": get_fiq_image_dataset(processor, "dress"),
            },
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        trainer.save_model()
        accelerator.end_training()


if __name__ == "__main__":
    CLI(main)
