import os
import sys
from typing import List

import fire
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset,  load_from_disk
import transformers
from transformers import Trainer
import torch.distributed as dist
import webdataset as wds
import torch.nn.functional as F
from torchvision import transforms
NIL_DATASET = True

from transformers import LlamaTokenizer, LlamaConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import set_seed
from transformers import BitsAndBytesConfig

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from transformers.utils import logging
from transformers.trainer_callback import TrainerCallback
logger = logging.get_logger(__name__)
from typing import List, Optional
from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches
from transformers import LlavaNextForConditionalGeneration
import torch
import torch.utils.checkpoint
from torch import nn

def pdist(A, B, squared = False, eps = 1e-12):
    D = A.pow(2).sum(1) + (-2) * B.mm(A.t())
    D = (B.pow(2).sum(1) + D.t()).clamp(min=eps)
    
    if not squared:
        D = D.sqrt()
        
    if torch.equal(A,B):
        D = D.clone()
        D[range(len(A)), range(len(A))] = 0
        
    return D


def relaxed_contrastive_loss(t_emb, s_emb, sigma=1, delta=1):
    s_emb = F.normalize(s_emb, p=2, dim=1)
    
    T_dist = pdist(t_emb, t_emb, False)
    dist_mean = T_dist.mean(1, keepdim=True)
    T_dist = T_dist / dist_mean
        
    with torch.no_grad():
        S_dist = pdist(s_emb, s_emb, False)
        P = torch.exp(-S_dist.pow(2) / sigma)
    
    pos_weight = P
    neg_weight = 1-P
    
    pull_losses = torch.relu(T_dist).pow(2) * pos_weight
    push_losses = torch.relu(delta - T_dist).pow(2) * neg_weight

    pull_losses = pull_losses[T_dist>0]
    push_losses = push_losses[T_dist>0]
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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
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
            if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
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
                        pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
                    ]
                    pixel_values = torch.cat(_pixel_values_list, dim=0)
                elif pixel_values.dim() != 4:
                    # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                    raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

                image_features = self.vision_tower(pixel_values, output_hidden_states=True)
                assert all(map(lambda x: x == image_num_patches[0], image_num_patches))
                res_last_hidden_vision = image_features.pooler_output[::image_num_patches[0]]
                selected_image_feature = image_features.hidden_states[vision_feature_layer]

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
                inputs_embeds, attention_mask, position_ids, labels = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )

            # pixel_values is not None but is empty ---> text only cases
            elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
                # there are no images
                pass

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

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

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)

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
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return res_last_hidden_vision, outputs.hidden_states
    
llama3_template = '''<|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'

class ForceTqdmUpdateCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # pdsh can't update tqdm, except warning
        if state.is_world_process_zero:
            if state.global_step % 5 == 0 or state.global_step < 20:
                logger.warning('')
@dataclass
class DataCollatorForSeq2SeqForNeg:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        _features = self.tokenizer.pad(
            {'input_ids': [feature['input_ids'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        _features['attention_mask'] = self.tokenizer.pad(
            {'input_ids': [feature['attention_mask'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )['input_ids']
        _features['labels'] = self.tokenizer.pad(
            {'input_ids': [feature['labels'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )['input_ids']
        features = _features


        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

from transformers.trainer_utils import has_length
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
)
from torch.utils.data import RandomSampler, SequentialSampler

class SentembTrainer(Trainer):
    force_tqdm_update = True
    fix_attention_mask = False

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.force_tqdm_update:
            self.add_callback(ForceTqdmUpdateCallback)

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        return RandomSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        clip_emb, hid_state = model(**inputs, output_hidden_states=True, return_dict=True)
        hid_state = hid_state[-1][:, -1, :]

        if dist.is_initialized():
            z1_list = [torch.zeros_like(clip_emb) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(hid_state) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z1_list, tensor=clip_emb.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=hid_state.contiguous())
            z1_list[dist.get_rank()] = clip_emb
            z2_list[dist.get_rank()] = hid_state
            clip_emb = torch.cat(z1_list, 0)
            hid_state = torch.cat(z2_list, 0)

        loss = relaxed_contrastive_loss(hid_state, clip_emb)
        return (loss, ) if return_outputs else loss

        clip_simmat = F.cosine_similarity(clip_emb.unsqueeze(1), clip_emb.unsqueeze(0), dim=-1)
        hid_simmat = F.cosine_similarity(hid_state.unsqueeze(1), hid_state.unsqueeze(0), dim=-1)

        bsize = clip_simmat.size(0)
        loss = ((clip_simmat - hid_simmat) ** 2).sum() / bsize

        return (loss, ) if return_outputs else loss


        if self.is_nli and self.use_neg_sentence:
            input_ids, labels, neg = inputs["input_ids"], inputs["labels"], inputs['attention_mask']
            pad_token_id = self.tokenizer.pad_token_id
            if self.fix_attention_mask:
                labels[labels < 0 ] = pad_token_id
                neg[neg < 0] = pad_token_id
            else:
                labels[labels < 0 ] = 0
                neg[neg < 0] = 0
            # padding tensor length
            mw = max(input_ids.size(1), labels.size(1), neg.size(1))

            pad_size = mw - labels.size(1)
            if pad_size > 0:
                label_pads = torch.zeros(labels.size(0), pad_size).cuda().long()
                label_pads.fill_(pad_token_id)
                labels = torch.cat([label_pads, labels], dim=1)
            pad_size = mw - input_ids.size(1)
            if pad_size > 0:
                input_pads = torch.zeros(input_ids.size(0), pad_size).cuda().long()
                input_pads.fill_(pad_token_id)
                input_ids = torch.cat([input_pads,
                                       input_ids], dim=1)
            pad_size = mw - neg.size(1)
            if pad_size > 0:
                neg_pads = torch.zeros(neg.size(0), pad_size).cuda().long()
                neg_pads.fill_(pad_token_id)
                neg = torch.cat([neg_pads,
                                 neg], dim=1)

            inputs['input_ids'] = torch.cat([input_ids, labels, neg], dim=0)
            if self.fix_attention_mask:
                inputs['attention_mask'] = (inputs['input_ids'] != pad_token_id).long()
            else:
                inputs['attention_mask'] = (inputs['input_ids'] > 0).long()
            del inputs['labels']
        elif self.is_nli:
            input_ids, labels = inputs["input_ids"], inputs["labels"]
            labels[labels < 0 ] = 0
            # padding tensor length
            if input_ids.size(1) > labels.size(1):
                pad_size = input_ids.size(1) - labels.size(1)
                labels = torch.cat([torch.zeros(labels.size(0), pad_size).cuda().long(), labels], dim=1)
            else:
                pad_size = labels.size(1) - input_ids.size(1)
                input_ids = torch.cat([torch.zeros(input_ids.size(0), pad_size).cuda().long(), input_ids], dim=1)
            inputs['input_ids'] = torch.cat([input_ids, labels], dim=0)
            inputs['attention_mask'] = (inputs['input_ids'] > 0).long()
            del inputs['labels']
        else:
            inputs['input_ids'] = torch.cat([inputs['input_ids'], inputs['input_ids']], dim=0)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], inputs['attention_mask']], dim=0)
            del inputs['labels']

        pooler_output = model(output_hidden_states=True, return_dict=True, **inputs).hidden_states[-1][:, -1, :]

        if self.use_neg_sentence:
            batch_size = pooler_output.size(0)//3
            pooler_output = torch.stack([pooler_output[:batch_size],
                                         pooler_output[batch_size:2*batch_size],
                                         pooler_output[2*batch_size:]], dim=1)
            z1, z2, z3 = pooler_output[:,0], pooler_output[:,1], pooler_output[:,2]
        else:
            batch_size = pooler_output.size(0)//2
            pooler_output = torch.stack([pooler_output[:batch_size], pooler_output[batch_size:]], dim=1)
            z1, z2 = pooler_output[:,0], pooler_output[:,1]
        loss_fct = nn.CrossEntropyLoss()

        if dist.is_initialized():
            if self.use_neg_sentence:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        if not hasattr(model, "sim"):
            self.sim = Similarity(temp=0.05)
        cos_sim = self.sim(z1.unsqueeze(1).float(), z2.unsqueeze(0).float())

        if self.use_neg_sentence:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(inputs['input_ids'].device)

        if self.use_neg_sentence:
            z3_weight = 0
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(input_ids.device)
            cos_sim = cos_sim + weights
        loss = loss_fct(cos_sim, labels)
        return (loss, pooler_output) if return_outputs else loss

def generate_sentemb_prompt(data_point, tokenizer, cutoff_len, template, prefix='input'):
    sp = f's{prefix}'
    if sp not in data_point:
        input = tokenizer(
            data_point[prefix],
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        input = tokenizer.decode(input['input_ids'])
        data_point[sp] = input
    else:
        input = data_point[sp]

    template = template.replace('_', ' ').replace('*sep+*', '')\
                                         .replace('*cls*', '').replace('\\n', '\n')
    return template.replace('*sent 0*', input).strip()

def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "data/nli_for_simcse.csv",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 256,
        micro_batch_size: int = 64,
        num_epochs: int = 1,
        learning_rate: float = 5e-4,
        cutoff_len: int = 32,
        # lora hyperparams
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve,
        is_sentemb: bool = False,
        mask_embedding_sentence_template: str = None,
        run_name: str = None,
        use_neg_sentence: bool = False,
        load_kbit: int = 4,
        save_steps: int = 100,
        seed: int = 42,
        deepspeed: str = None,
        logging_steps: int = 10,
        grad_checkpoint: bool = False,
        fix_attention_mask: bool = False,
        set_pad_to_unk: bool = False,
        bf16: bool = False,
        not_eol: bool = False,
        org_attn: bool = False,
        *arg,
        **kwarg
):
    # set NCCL_DEBUG

    global NIL_DATASET
    NIL_DATASET = True


    group_by_length = False
    train_on_inputs = False
    cutoff_len = 32

    assert load_kbit in [4, 8, 16]

    run_name = output_dir

    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    #if ddp and False:
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

        torch.distributed.init_process_group("nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        torch.cuda.set_device(device)

    set_seed(seed)

    config = None

    dtype = torch.float16 if load_kbit == 16 else torch.float32
    if bf16:
        dtype = torch.bfloat16

    if 'Phi-3' not in base_model:
        from accelerate import Accelerator
        accelerator = Accelerator()
        #device = accelerator.device
        with accelerator.main_process_first():
            base_llm_model = base_model.split('/')[-1] + '-llm'
            base_llm_model = os.path.join('models', base_llm_model)
            base_llm_model = base_llm_model.strip('-')
            if not os.path.exists(base_llm_model):
                LlavaNextForConditionalGeneration.from_pretrained(
                    base_model,
                    device_map='cpu',
                ).language_model.save_pretrained(base_llm_model)

        if load_kbit == 4:
            assert load_kbit == 4
            model = HackLlavaNextForConditionalGeneration.from_pretrained(
                base_model,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                ),
                torch_dtype=torch.bfloat16 if bf16 else torch.float16,
                device_map=device_map,
                attn_implementation='eager' if org_attn else None,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_llm_model,
                load_in_8bit=load_kbit == 8,
                load_in_4bit=load_kbit == 4,
                torch_dtype=torch.bfloat16 if bf16 else torch.float16,
                device_map=device_map,
                attn_implementation='eager' if org_attn else None,
            )
    elif load_kbit == 4:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
            config=config,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            _attn_implementation='eager' if 'phi3' in base_model else None,
            trust_remote_code=True,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_kbit == 8 ,
            torch_dtype=dtype,
            device_map=device_map,
        )


    if 'llama-3' in base_model:
        tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer.padding = True
    elif 'llava' in base_model:
        from transformers import LlavaNextProcessor
        if base_model == "llava-hf/llava-v1.6-mistral-7b-hf":
            # bug in new vision of tokenizer
            tokenizer = LlavaNextProcessor.from_pretrained(base_model, revision='a1d521368f8d353afa4da2ed2bb1bf646ef1ff5f').tokenizer
        else:
            tokenizer = LlavaNextProcessor.from_pretrained(base_model).tokenizer
    elif 'Phi-3' in base_model:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        tokenizer.padding = True
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)

        if tokenizer.bos_token_id == 0:
            # fix bos token id
            tokenizer.bos_token_id = 1
            tokenizer.eos_token = '</s>'

        tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        tokenizer.padding_side = "left"  # Allow batched inference

    if set_pad_to_unk:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    if 'llama-3' in base_model:
        mask_embedding_sentence_template = llama3_template.format(mask_embedding_sentence_template)

    if not_eol:
        mask_embedding_sentence_template = '*sent_0*'
    print(mask_embedding_sentence_template)
    def tokenize(prompt, add_eos_token=True, label_prompt=None, neg_prompt=None):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        if label_prompt:
            label_result = tokenizer(
                label_prompt,
                padding=False,
                return_tensors=None,
            )
            result["labels"] = label_result["input_ids"]
            if neg_prompt:
                neg_result = tokenizer(
                    neg_prompt,
                    padding=False,
                    return_tensors=None,
                )
                result["attention_mask"] = neg_result["input_ids"]
        else:
            result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if NIL_DATASET:
            data_point['input'] = data_point['sent0']
            data_point['output'] = data_point['sent1']
            if use_neg_sentence:
                data_point['neg'] = data_point['hard_neg']

        full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                              mask_embedding_sentence_template,
                                              prefix='input')
        if NIL_DATASET:
            pos_full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                                      mask_embedding_sentence_template,
                                                      prefix='output')
            if use_neg_sentence:
                neg_full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len,
                                                          mask_embedding_sentence_template,
                                                          prefix="neg")

        tokenized_full_prompt = tokenize(full_prompt, False,
                                         label_prompt=None if not NIL_DATASET else pos_full_prompt,
                                         neg_prompt=neg_full_prompt if NIL_DATASET and use_neg_sentence else None)
        if not train_on_inputs and not NIL_DATASET:
            user_prompt = generate_sentemb_prompt({**data_point, "output": ""}, tokenizer, cutoff_len,
                                                  mask_embedding_sentence_template,
                                                  prefix='input')
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if grad_checkpoint:
        model.enable_input_require_grads()

    if load_kbit == 4:
        if 'Phi-3' in base_model:
            target_modules = [
                [f'model.layers.{i}.mlp.gate_up_proj',
                 f'model.layers.{i}.mlp.down_proj',
                 f'model.layers.{i}.self_attn.o_proj',
                 f'model.layers.{i}.self_attn.qkv_proj' ] for i in range(32)
            ]
            target_modules = sum(target_modules, [])
            print(target_modules)
        else:
            model = prepare_model_for_kbit_training(model)
            def find_all_linear_names(model):
                cls = bnb.nn.Linear4bit
                lora_module_names = set()
                for name, module in model.named_modules():
                    if isinstance(module, cls):
                        names = name.split('.')
                        lora_module_names.add(names[0] if len(names) == 1 else names[-1])

                if 'lm_head' in lora_module_names: # needed for 16-bit
                    lora_module_names.remove('lm_head')
                return list(lora_module_names)
            target_modules = find_all_linear_names(model)
            target_modules = "|".join(end for end in target_modules)
            target_modules = f"^language_model.*({target_modules})$"
            print(target_modules)

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    else:
        if load_kbit == 8:
            model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    if 'csv' in data_path:
        data = load_dataset("csv", data_files=data_path)
    elif os.path.isdir(data_path):
        data = load_from_disk(data_path)
    else:
        data = load_dataset("json", data_files=data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    from transformers import LlavaNextProcessor
    transform = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
    transform.tokenizer = tokenizer
    transform.tokenizer.add_tokens('<image>')
    transform.tokenizer.pad_token_id = transform.tokenizer.eos_token_id
    transform.tokenizer.padding_side = "left"
    transform.tokenizer.padding = True
    model.config.image_token_index = 128256

    def preprocess(x):
        res = transform(['<|start_header_id|>user<|end_header_id|>\n\n<image>\nSummary above image in one word: <|efot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'], x[0], return_tensors="pt", padding=True)
        for reskey in res:
            res[reskey] = res[reskey].squeeze_(0)
        return res
    
    dpth = os.path.expanduser("~/dataset/cc3m/{00000..00331}.tar")
    dlen = 200_000
    train_data = wds.WebDataset(dpth, shardshuffle=True, detshuffle=True, seed=seed, nodesplitter=wds.shardlists.split_by_node).with_length(dlen).with_epoch(dlen).decode('pil').to_tuple("jpg").map(preprocess)

    DC_FUN = DataCollatorForSeq2SeqForNeg if NIL_DATASET and use_neg_sentence else transformers.DataCollatorForSeq2Seq
    data_collator = DC_FUN(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        #tokenizer, return_tensors="pt", padding=True
    )

    trainer = SentembTrainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True if not bf16 else False,
            bf16=bf16,
            logging_steps=logging_steps,
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=100,
            load_best_model_at_end=False,
            #ddp_find_unused_parameters=False if ddp else None,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            run_name=run_name,
            report_to=None,
            deepspeed=deepspeed,
            gradient_checkpointing=grad_checkpoint,
        ),
        # data_collator=data_collator,
    )
    trainer.tokenizer = tokenizer
    trainer.is_nli = NIL_DATASET
    trainer.use_neg_sentence = use_neg_sentence
    trainer.fix_attention_mask = fix_attention_mask
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
