# Adapted from https://github.com/NVIDIA/NeMo/blob/v2.0.0rc0/nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_dataset.py
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import re
from typing import List, Mapping, Optional

import datasets
import numpy as np
import torch
from datasets import load_dataset

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset, OnlineSampleMapping
from nemo.core.classes import Dataset
from nemo.utils import logging

import flatflow.torch.utils.data


__all__ = ['FlatflowMegatronDataset']


class FlatflowMegatronDataset(flatflow.torch.utils.data.Dataset):
    """
    A dataset class implementing interfaces for TEDotProductAttention and Flatflow scheduler. 

    1. Flatflow scheduler requires sizes of all items, to yield indices of scheduled samples in order.
    2. TEDotProductAttention requires cu_seqlens, to utilize `flash_attn_varlen_func`.

    It processes a jsonl data with formatting and tokenization. And it defines `collate_fn` to 
    concatenate each batch sampled by the Flatflow scheduler, which will be used inside a data loader.

    * Many parts are adopted from megatron's GPTSFTDataset and GPTSFTPackedDataset, then modified accordingly.
    """
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        pad_seq_length_to_mult: int = 16,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        max_num_samples: int = None,
        seed: int = 1234,
        label_key: str = "answer",
        answer_only_loss: bool = True,
        truncation_field: str = "text",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        index_mapping_dir: str = None,
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        memmap_workers: Optional[int] = None,
        hf_dataset: bool = False,
        truncation_method: str = 'right',
        special_tokens: Optional[Mapping[str, str]] = None,  # special tokens, a dictory of {token_type: token}
        is_test: bool = False,
        output_original_text: bool = False,
        ceil_to_power_2: bool = False,
        get_attention_mask_from_fusion: bool = False,
    ):
        np.random.seed(kwargs.get('seed', 1234))
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.pad_seq_length_to_mult = pad_seq_length_to_mult
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.sep_id = sep_id
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.label_key = label_key
        self.answer_only_loss = answer_only_loss
        self.truncation_fields = truncation_field.split(',')
        self.pad_to_max_length = pad_to_max_length
        self.index_mapping_dir = index_mapping_dir
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.memmap_workers = memmap_workers
        self.hf_dataset = hf_dataset
        self.truncation_method = truncation_method
        self.is_test = is_test
        self.output_original_text = output_original_text
        self.ceil_to_power_2 = ceil_to_power_2
        self.get_attention_mask_from_fusion = get_attention_mask_from_fusion

        if special_tokens is None:
            self.special_tokens = {
                "system_turn_start": "<extra_id_0>",
                "turn_start": "<extra_id_1>",
                "label_start": "<extra_id_2>",
                "end_of_turn": "\n",
                "end_of_name": "\n",
            }
        else:
            self.special_tokens = special_tokens

        self._load_dataset()
        # Validate prompt template
        self._maybe_validate_prompt_template()

        # Will be None after this call if `max_num_samples` is None
        self._build_samples_mapping()
        self._process_dataset()

    def _load_dataset(self):
        if self.hf_dataset:
            self.indexed_dataset = load_dataset(
                'json',
                data_files=self.file_path,
                cache_dir=self.index_mapping_dir,
                num_proc=self.memmap_workers,
                split='train',
            )
        else:
            self.indexed_dataset = JSONLMemMapDataset(
                dataset_paths=[self.file_path],
                tokenizer=None,
                header_lines=0,
                index_mapping_dir=self.index_mapping_dir,
                workers=self.memmap_workers,
            )
         
    def _maybe_validate_prompt_template(self):
        assert (
            self.prompt_template is not None
        ), f'we need prompt_template to combine contexts and label {self.label_key}'
        # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
        self.prompt_template = self.prompt_template.encode('utf-8').decode('unicode_escape')
        self.prompt_template_keys = re.findall(r'{(.*?)}', self.prompt_template)

        label_placeholder = f'{{{self.label_key}}}'
        assert (
            self.prompt_template[-len(label_placeholder) :] == label_placeholder
        ), f'{label_placeholder} must be at the end of prompt_template.'

        # Legacy checkpoints has self.truncation_fields = ['context'] and self.prompt_template_keys = ['input', 'output']
        if self.prompt_template_keys[0] == 'input' and self.truncation_fields[0] == 'context':
            self.truncation_fields[0] = self.prompt_template_keys[0]

        assert set(self.truncation_fields).issubset(
            self.prompt_template_keys
        ), f'truncation_fields {self.truncation_fields} must in {self.prompt_template_keys}'
    
    def _process_dataset(self):
        """
        Process the loaded dataset.

        Apply processing of `GPTSFTDataset` first, then further prepare the examples for teacher-forcing
        scheme in advance, not later inside `collate_fn` like `GPTSFTDataset` where you should substract 1
        every time.

        This procedure is highly dependent on `GPTSFTDataset._process_example`, check its update carefully.
        """
        # TODO: multiprocess it for speed-up.
        processed_dataset = []
        for example in self.indexed_dataset:
            example = self._process_example(example)
            example['labels'] = example['input_ids'][1:]
            example['input_ids'] = example['input_ids'][:-1]
            example['token_count'] -= 1
            processed_dataset.append(example)

        self.indexed_dataset = processed_dataset

    def _build_samples_mapping(self):
        """Flatflow doesn't use a separate sample mapping, but a scheduler instead."""
        self.samples_mapping = None

    def __sizeof__(self, index: int) -> int:
        """Return the relative size of element for scheduling."""
        return self.indexed_dataset[index]['token_count']

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, index):
        item = self.indexed_dataset[index]
        input_ids = item['input_ids']
        labels = item['labels']
        seqlen = item['token_count']
        loss_mask = self._build_loss_mask(item) if index >= 0 else [0] * seqlen
        return {'input_ids': input_ids, 'labels': labels, 'loss_mask': loss_mask, 'seqlen': seqlen}

    def collate_fn(self, batch):
        """Process & concatenate the batch of samples. Padding is not applied at all."""
        input_ids = [np.concatenate([item['input_ids'] for item in batch])]
        labels = [np.concatenate([item['labels'] for item in batch])]
        loss_mask = [np.concatenate([item['loss_mask'] for item in batch])]
        position_ids = [np.concatenate([list(range(item['seqlen'])) for item in batch])]
        token_count = [input_ids[0].shape[0]]

        if input_ids[0].shape[0] != position_ids[0].shape[0]: 
            raise ValueError("Dataset problem: input_ids and position_ids lengths don't match")

        # Process cu_seqlens-related values.
        seqlens = np.array([[item['seqlen'] for item in batch]])
        # Here, concatenating -1 at the end is for `cu_seqlens_argmin`-related operations in `MegatronGPTModel`.
        # TBH, padding then removing -1(s) at the end later look unreasonable and error-prone.
        # See a related PR: https://github.com/NVIDIA/NeMo/pull/8108/files
        cu_seqlens = np.concatenate([[[0]], seqlens.cumsum(axis=1), [[-1]]], axis=1)
        cu_seqlens_argmin = np.argmin(cu_seqlens, axis=1, keepdims=True)
        max_seqlen = seqlens.max(keepdims=True)

        processed_batch = {
            'tokens': torch.LongTensor(input_ids),
            'labels': torch.LongTensor(labels),
            'loss_mask': torch.LongTensor(loss_mask),
            'position_ids': torch.LongTensor(position_ids),
            'token_count': token_count,
            'attention_mask': torch.LongTensor([1] * len(input_ids)),  # no attention mask is needed for packed seq
            'cu_seqlens': torch.IntTensor(cu_seqlens),  # cu_seqlens must be in dtype torch.int32
            'cu_seqlens_argmin': torch.IntTensor(cu_seqlens_argmin),
            'max_seqlen': torch.IntTensor(max_seqlen),
        }
        return processed_batch
