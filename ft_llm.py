import os
import sys
from typing import List, Optional

import bitsandbytes as bnb
import fire
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import webdataset as wds
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from torch.utils.data import RandomSampler
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    Trainer,
    set_seed,
)
from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import has_length
from transformers.utils import logging

logger = logging.get_logger(__name__)
llama3_template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"


def pdist(A, B, squared=False, eps=1e-12):
    D = A.pow(2).sum(1) + (-2) * B.mm(A.t())
    D = (B.pow(2).sum(1) + D.t()).clamp(min=eps)

    if not squared:
        D = D.sqrt()

    if torch.equal(A, B):
        D = D.clone()
        D[range(len(A)), range(len(A))] = 0

    return D


def relaxed_contrastive_loss(t_emb, s_emb, sigma=0.75, delta=1):
    s_emb = F.normalize(s_emb, p=2, dim=1)

    T_dist = pdist(t_emb, t_emb, False)
    dist_mean = T_dist.mean(1, keepdim=True)
    T_dist = T_dist / dist_mean

    with torch.no_grad():
        S_dist = pdist(s_emb, s_emb, False)
        P = torch.exp(-S_dist.pow(2) / sigma)

    pos_weight = P
    neg_weight = 1 - P

    pull_losses = torch.relu(T_dist).pow(2) * pos_weight
    push_losses = torch.relu(delta - T_dist).pow(2) * neg_weight

    pull_losses = pull_losses[T_dist > 0]
    push_losses = push_losses[T_dist > 0]
    loss = (pull_losses.sum() + push_losses.sum()) / len(t_emb)

    return loss


class HackLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
            for_inputs_embeds_ids = input_ids.clone()
            for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0
            inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

            # 2. Merge text and images
            if (
                pixel_values is not None
                and input_ids.shape[1] != 1
                and pixel_values.size(0) > 0
            ):
                # ! infer image_num_patches from image_sizes
                image_num_patches = [
                    image_size_to_num_patches(
                        image_size=imsize,
                        grid_pinpoints=self.config.image_grid_pinpoints,
                        patch_size=self.config.vision_config.image_size,
                    )
                    for imsize in image_sizes
                ]
                # figure out if pixel_values is concatenated or stacked
                if pixel_values.dim() == 5:
                    # stacking when input is (batch_size, num_patches, num_channels, height, width)
                    _pixel_values_list = [
                        pix_val[:num_patch]
                        for pix_val, num_patch in zip(pixel_values, image_num_patches)
                    ]
                    pixel_values = torch.cat(_pixel_values_list, dim=0)
                elif pixel_values.dim() != 4:
                    # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                    raise ValueError(
                        f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions"
                    )

                image_features = self.vision_tower(
                    pixel_values, output_hidden_states=True
                )
                assert all(map(lambda x: x == image_num_patches[0], image_num_patches))
                res_last_hidden_vision = image_features.pooler_output[
                    :: image_num_patches[0]
                ]
                selected_image_feature = image_features.hidden_states[
                    vision_feature_layer
                ]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature

                image_features = self.multi_modal_projector(selected_image_feature)

                image_features = torch.split(image_features, image_num_patches, dim=0)

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"

                image_features, feature_lens = self.pack_image_features(
                    image_features,
                    image_sizes,
                    image_newline=self.image_newline,
                )

                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, position_ids, labels = (
                    self._merge_input_ids_with_image_features(
                        image_features,
                        feature_lens,
                        inputs_embeds,
                        input_ids,
                        attention_mask,
                        position_ids,
                        labels=labels,
                    )
                )

            # pixel_values is not None but is empty ---> text only cases
            elif (
                pixel_values is not None
                and input_ids.shape[1] != 1
                and pixel_values.size(0) == 0
            ):
                # there are no images
                pass

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif (
                past_key_values is not None
                and pixel_values is not None
                and input_ids.shape[1] == 1
            ):
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(
                    first_layer_past_key_value.float().sum(-2) == 0
                )

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat(
                    (extended_attention_mask, attention_mask[:, -target_length:]), dim=1
                )

                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return res_last_hidden_vision, outputs.hidden_states


class ForceTqdmUpdateCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # pdsh can't update tqdm, except warning
        if state.is_world_process_zero:
            if state.global_step % 5 == 0 or state.global_step < 20:
                logger.warning("")


class SentembTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        self.add_callback(ForceTqdmUpdateCallback)
        return RandomSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        clip_emb, hid_state = model(
            **inputs, output_hidden_states=True, return_dict=True
        )
        hid_state = hid_state[-1][:, -1, :]

        if dist.is_initialized():
            z1_list = [torch.zeros_like(clip_emb) for _ in range(dist.get_world_size())]
            z2_list = [
                torch.zeros_like(hid_state) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(tensor_list=z1_list, tensor=clip_emb.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=hid_state.contiguous())
            z1_list[dist.get_rank()] = clip_emb
            z2_list[dist.get_rank()] = hid_state
            clip_emb = torch.cat(z1_list, 0)
            hid_state = torch.cat(z2_list, 0)

        loss = relaxed_contrastive_loss(hid_state, clip_emb)
        return (loss,) if return_outputs else loss


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 256,
    micro_batch_size: int = 64,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
    # lora hyperparams
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # llm hyperparams
    seed: int = 42,
    deepspeed: str = None,
    bf16: bool = False,
    *arg,
    **kwarg,
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

        torch.distributed.init_process_group("nccl")
        rank, world_size = (
            torch.distributed.get_rank(),
            torch.distributed.get_world_size(),
        )
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        torch.cuda.set_device(device)

    set_seed(seed)

    dtype = torch.bfloat16 if bf16 else torch.float16
    model = HackLlavaNextForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model)

    def find_all_linear_names(model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
        return list(lora_module_names)

    target_modules = find_all_linear_names(model)
    target_modules = "|".join(end for end in target_modules)
    target_modules = f"^language_model.*({target_modules})$"
    print(target_modules)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    transform = get_transform(model)
    train_data = get_dataset(seed, transform)

    trainer = SentembTrainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            bf16=bf16,
            fp16=not bf16,
            ddp_find_unused_parameters=False if ddp else None,
            deepspeed=deepspeed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            learning_rate=learning_rate,
            logging_steps=1,
            num_train_epochs=num_epochs,
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            run_name=output_dir,
            save_steps=50,
            save_total_limit=100,
            warmup_steps=100,
        ),
    )
    trainer.tokenizer = transform.tokenizer

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained(output_dir)
    torch.distributed.destroy_process_group()


def get_dataset(seed, transform):
    def preprocess(x):
        res = transform(
            [llama3_template.format("<image>\nSummary above image in one word: ")],
            x[0],
            return_tensors="pt",
            padding=True,
        )
        for reskey in res:
            res[reskey] = res[reskey].squeeze_(0)
        return res

    dpth = os.path.expanduser("~/dataset/cc3m/{00000..00331}.tar")
    dlen = 200_000
    train_data = (
        wds.WebDataset(
            dpth,
            shardshuffle=True,
            detshuffle=True,
            seed=seed,
            nodesplitter=wds.shardlists.split_by_node,
        )
        .with_length(dlen)
        .with_epoch(dlen)
        .decode("pil")
        .to_tuple("jpg")
        .map(preprocess)
    )

    return train_data


def get_transform(model):
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
    tokenizer.add_tokens("<image>")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.padding = True
    model.config.image_token_index = 128256

    transform = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    transform.tokenizer = tokenizer
    return transform


if __name__ == "__main__":
    fire.Fire(train)
