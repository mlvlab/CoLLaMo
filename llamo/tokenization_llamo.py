import re

import torch
from transformers import LlamaTokenizer, PreTrainedTokenizerFast

from pipeline.data_utils.constants import IGNORE_INDEX, MEDIA_TOKENS
from utils.logging import get_logger

_INFINITE = int(1e12)  # infinite token length for no-truncation

logger = get_logger()


def _pad_trunc(
    x,
    padding,
    padding_side,
    pad_value,
    max_length,
):
    """Pad and truncate sequences to the same length

    Args:
        x (list[list[int]])
        padding ("longest" or "max_length")
        padding_side ("left" or "right")
        pad_value (int)
        max_length (int or None): if padding == "max_length", max_length should be given.
    """
    assert padding in ["longest", "max_length"]
    assert padding_side in ["left", "right"]

    lengths = [len(sample) for sample in x]
    if padding == "longest":
        max_length = max(lengths)

    new_x = []
    for sample, length in zip(x, lengths):
        if torch.is_tensor(sample):
            sample = sample.tolist()

        if length >= max_length:
            new_x.append(sample[:max_length])
            continue

        padding_size = max_length - length
        pads = [pad_value] * padding_size
        if padding_side == "right":
            new_x.append(sample + pads)
        else:
            new_x.append(pads + sample)

    return torch.as_tensor(new_x, dtype=torch.long)


class LLaMoTokenizer(LlamaTokenizer):
    def selfies_setup(self):
        alphabet_set = set()
        with open("./data/selfies_alphabet.txt", "r") as f:
            alphabet = f.readlines()
            for a in alphabet:
                if "[" not in a:
                    continue
                alphabet_set.add(a.strip())
        self.add_tokens(sorted(list(alphabet_set)))
        special_tokens_dict = {"additional_special_tokens": ["<bom>", "<eom>"]}
        self.add_special_tokens(
            special_tokens_dict, replace_additional_special_tokens=False
        )

    def mllm_setup(self, num_graph_tokens: int):
        # if self.pad_token is None:
        #     logger.warning(f"Tokenizer {self.__class__} has no pad_token. Use unk_token instead.")
        #     self.pad_token = self.unk_token

        self.num_graph_tokens = num_graph_tokens

        # Currently we only support the image modality for media modality.
        self.media_tokens = {
            k: -int(i + 1) for i, k in enumerate(MEDIA_TOKENS["graph"])
        }
        self.media_lengths = {
            MEDIA_TOKENS["graph"][0]: num_graph_tokens,
            MEDIA_TOKENS["graph"][1]: num_graph_tokens,
        }  # token lengths

    def encode_prompt(self, prompt, max_length, no_eos=False):
        """Tokenize prompt which consists of image-text or text only, with role tokens.
        Role pattern is "AI: " or "Human: ".

        Args:
            prompt
            max_length (int or None): here, max_length is used for truncation.
                If max_length is None, no truncation is applied.
            no_eos: if True, eos token is not added at the end of the prompt.
                Note that eos token is still used for end-of-AI-turn token even no_eos=True.
        """
        max_length = (
            max_length or _INFINITE
        )  # if None, set to infinite for no-truncation

        # output enc_chunk
        enc_chunk = [self.bos_token_id]
        label_chunk = [0]

        pattern = "|".join(map(re.escape, list(self.media_tokens.keys())))

        for idx, m in enumerate(prompt):
            if len(enc_chunk) > max_length:
                break

            if m["role"] == "system":
                curr_message = "[INST] <<SYS>>\n" + m["content"] + "\n<</SYS>>\n\n"

            elif m["role"] == "user":
                if idx == 1:
                    curr_message += m["content"] + " [/INST]"
                else:
                    curr_message = "<s>[INST] " + m["content"] + " [/INST]"
                chunk_strs = re.split(f"({pattern})", curr_message)
                chunk_strs = [x.strip() for x in chunk_strs if len(x) > 0]
                for _, chunk_str in enumerate(chunk_strs):
                    if chunk_str in self.media_tokens:
                        enc_chunk += [
                            self.media_tokens[chunk_str]
                        ] * self.media_lengths[chunk_str]
                        label_chunk += [0] * self.media_lengths[chunk_str]
                    else:
                        curr_chunk = self(chunk_str, add_special_tokens=False)[
                            "input_ids"
                        ]
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)
            elif m["role"] == "assistant":
                curr_message = m["content"] + "</s>"
                curr_chunk = self(curr_message, add_special_tokens=False)["input_ids"]
                enc_chunk += curr_chunk
                label_chunk += [1] * len(curr_chunk)
            else:
                raise ValueError("Role should be either system, user, or assistant")

        if no_eos and enc_chunk[-1] == self.eos_token_id:
            # the last token can be != eos_token_id; when the prompt is ended with `AI: `.
            # in this case, there is no AI-answer, thus, no eos token is added.
            enc_chunk = enc_chunk[:-1]
            label_chunk = label_chunk[:-1]

        enc_chunk = enc_chunk[: max_length + 1]
        label_chunk = label_chunk[: max_length + 1]
        L = len(enc_chunk)
        assert L == len(label_chunk)

        input_ids = torch.as_tensor(enc_chunk, dtype=torch.long)
        loss_mask = torch.as_tensor(label_chunk, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Label
        labels = input_ids.clone()
        labels[loss_mask != 1] = IGNORE_INDEX

        # The length of input_ids (L) includes <bos> and <eos> tokens.
        # Since max_length does not include <bos> token, L <= max_length + 1
        assert L <= max_length + 1

        return {
            "input_ids": input_ids,  # [L]
            "labels": labels,  # [L]
            "seq_length": L,  # int
            "attention_mask": attention_mask,  # [L]
        }

    def batch_encode_prompt(
        self,
        prompts,
        padding="longest",
        padding_side="right",
        max_length=None,
        no_eos=False,
    ):
        """Batch encode prompts, pad/truncate to the same length, and collate them.
        Args:
            prompts (list[str])
            padding ("longest" or "max_length")
            padding_side ("left" or "right")
            pad_value (int)
            max_length (int or None): if padding == "max_length", max_length should be given
        """
        batch = [self.encode_prompt(prompt, max_length, no_eos) for prompt in prompts]
        batch = self.batch_collate_pad(batch, padding, padding_side, max_length)

        return batch

    def batch_collate_pad(
        self,
        batch,
        padding,
        padding_side,
        max_length,
        is_eval=False,
    ):
        """Collate batch and pad/truncate to the same length
        Args:
            batch
            padding ("longest" or "max_length")
            padding_side ("left" or "right")
            pad_value (int)
            max_length (int or None): if padding == "max_length", max_length should be given
        """
        if padding == "max_length":
            assert max_length is not None, (
                "max_length should be given if padding == 'max_length'"
            )
        else:
            # if padding == 'longest' and max_length is None, set to infinite for no-truncation
            max_length = max_length or _INFINITE

        label_pad_value = IGNORE_INDEX
        input_padding_side = padding_side
        label_padding_side = padding_side
        if is_eval:
            label_pad_value = self.pad_token_id
            input_padding_side = "left"
            label_padding_side = "right"

        input_ids = [sample["input_ids"] for sample in batch]
        labels = [sample["labels"] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        seq_length = [sample["seq_length"] for sample in batch]

        # max_length + 1 for bos_token
        input_ids = _pad_trunc(
            input_ids, padding, input_padding_side, self.pad_token_id, max_length + 1
        )
        labels = _pad_trunc(
            labels, padding, label_padding_side, label_pad_value, max_length + 1
        )
        attention_mask = _pad_trunc(
            attention_mask, padding, input_padding_side, 0, max_length + 1
        )
        seq_length = torch.as_tensor(seq_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "seq_length": seq_length,
        }


class LLaMoTokenizerforLlama3(PreTrainedTokenizerFast):
    def selfies_setup(self):
        alphabet_set = set()
        with open("./data/selfies_alphabet.txt", "r") as f:
            alphabet = f.readlines()
            for a in alphabet:
                if "[" not in a:
                    continue
                alphabet_set.add(a.strip())
        self.add_tokens(sorted(list(alphabet_set)))
        special_tokens_dict = {"additional_special_tokens": ["<bom>", "<eom>"]}
        self.add_special_tokens(
            special_tokens_dict, replace_additional_special_tokens=False
        )

    def mllm_setup(self, num_graph_tokens: int):
        # if self.pad_token is None:
        #     logger.warning(f"Tokenizer {self.__class__} has no pad_token. Use unk_token instead.")
        #     self.pad_token = self.unk_token

        self.num_graph_tokens = num_graph_tokens

        # Currently we only support the image modality for media modality.
        self.media_tokens = {
            k: -int(i + 1) for i, k in enumerate(MEDIA_TOKENS["graph"])
        }
        self.media_lengths = {
            MEDIA_TOKENS["graph"][0]: num_graph_tokens,
            MEDIA_TOKENS["graph"][1]: num_graph_tokens,
        }  # token lengths

    def encode_prompt(self, prompt, max_length, no_eos=False):
        """Tokenize prompt which consists of image-text or text only, with role tokens.
        Role pattern is "AI: " or "Human: ".

        Args:
            prompt
            max_length (int or None): here, max_length is used for truncation.
                If max_length is None, no truncation is applied.
            no_eos: if True, eos token is not added at the end of the prompt.
                Note that eos token is still used for end-of-AI-turn token even no_eos=True.
        """
        max_length = (
            max_length or _INFINITE
        )  # if None, set to infinite for no-truncation

        # output enc_chunk
        enc_chunk = [self.bos_token_id]
        label_chunk = [0]

        pattern = "|".join(map(re.escape, list(self.media_tokens.keys())))

        for idx, m in enumerate(prompt):
            if len(enc_chunk) > max_length:
                break

            if m["role"] == "system":
                curr_message = (
                    "<|start_header_id|>system<|end_header_id|>\n\n"
                    + m["content"]
                    + "<|eot_id|>"
                )
                curr_chunk = self(curr_message, add_special_tokens=False)["input_ids"]
                enc_chunk += curr_chunk
                label_chunk += [0] * len(curr_chunk)

            elif m["role"] == "user":
                curr_message = (
                    "<|start_header_id|>user<|end_header_id|>\n\n"
                    + m["content"]
                    + "<|eot_id|>"
                )
                chunk_strs = re.split(f"({pattern})", curr_message)
                chunk_strs = [x.strip() for x in chunk_strs if len(x) > 0]
                for _, chunk_str in enumerate(chunk_strs):
                    if chunk_str in self.media_tokens:
                        enc_chunk += [
                            self.media_tokens[chunk_str]
                        ] * self.media_lengths[chunk_str]
                        label_chunk += [0] * self.media_lengths[chunk_str]
                    else:
                        curr_chunk = self(chunk_str, add_special_tokens=False)[
                            "input_ids"
                        ]
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)

            elif m["role"] == "assistant":
                curr_message = (
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    + m["content"]
                    + "<|eot_id|>"
                )
                curr_chunk = self(curr_message, add_special_tokens=False)["input_ids"]
                enc_chunk += curr_chunk
                label_chunk += [0] * 4 + [1] * (len(curr_chunk) - 4)
            else:
                raise ValueError("Role should be either system, user, or assistant")

        enc_chunk += self(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )["input_ids"]
        label_chunk += [0] * 4

        if no_eos and enc_chunk[-1] == self.eos_token_id:
            # the last token can be != eos_token_id; when the prompt is ended with `AI: `.
            # in this case, there is no AI-answer, thus, no eos token is added.
            enc_chunk = enc_chunk[:-1]
            label_chunk = label_chunk[:-1]

        enc_chunk = enc_chunk[: max_length + 1]
        label_chunk = label_chunk[: max_length + 1]
        L = len(enc_chunk)
        assert L == len(label_chunk)

        input_ids = torch.as_tensor(enc_chunk, dtype=torch.long)
        loss_mask = torch.as_tensor(label_chunk, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Label
        labels = input_ids.clone()
        labels[loss_mask != 1] = IGNORE_INDEX

        # The length of input_ids (L) includes <bos> and <eos> tokens.
        # Since max_length does not include <bos> token, L <= max_length + 1
        assert L <= max_length + 1

        return {
            "input_ids": input_ids,  # [L]
            "labels": labels,  # [L]
            "seq_length": L,  # int
            "attention_mask": attention_mask,  # [L]
        }

    def batch_encode_prompt(
        self,
        prompts,
        padding="longest",
        padding_side="right",
        max_length=None,
        no_eos=False,
    ):
        """Batch encode prompts, pad/truncate to the same length, and collate them.
        Args:
            prompts (list[str])
            padding ("longest" or "max_length")
            padding_side ("left" or "right")
            pad_value (int)
            max_length (int or None): if padding == "max_length", max_length should be given
        """
        batch = [self.encode_prompt(prompt, max_length, no_eos) for prompt in prompts]
        batch = self.batch_collate_pad(batch, padding, padding_side, max_length)

        return batch

    def batch_collate_pad(
        self,
        batch,
        padding,
        padding_side,
        max_length,
        is_eval=False,
    ):
        """Collate batch and pad/truncate to the same length
        Args:
            batch
            padding ("longest" or "max_length")
            padding_side ("left" or "right")
            pad_value (int)
            max_length (int or None): if padding == "max_length", max_length should be given
        """
        if padding == "max_length":
            assert max_length is not None, (
                "max_length should be given if padding == 'max_length'"
            )
        else:
            # if padding == 'longest' and max_length is None, set to infinite for no-truncation
            max_length = max_length or _INFINITE

        label_pad_value = IGNORE_INDEX
        input_padding_side = padding_side
        label_padding_side = padding_side
        if is_eval:
            label_pad_value = self.pad_token_id
            input_padding_side = "left"
            label_padding_side = "right"

        input_ids = [sample["input_ids"] for sample in batch]
        labels = [sample["labels"] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        seq_length = [sample["seq_length"] for sample in batch]

        # max_length + 1 for bos_token
        input_ids = _pad_trunc(
            input_ids, padding, input_padding_side, self.pad_token_id, max_length + 1
        )
        labels = _pad_trunc(
            labels, padding, label_padding_side, label_pad_value, max_length + 1
        )
        attention_mask = _pad_trunc(
            attention_mask, padding, input_padding_side, 0, max_length + 1
        )
        seq_length = torch.as_tensor(seq_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "seq_length": seq_length,
        }


class LLaMoTokenizerforFalcon(PreTrainedTokenizerFast):
    def selfies_setup(self):
        alphabet_set = set()
        with open("./data/selfies_alphabet.txt", "r") as f:
            alphabet = f.readlines()
            for a in alphabet:
                if "[" not in a:
                    continue
                alphabet_set.add(a.strip())
        self.add_tokens(sorted(list(alphabet_set)))
        special_tokens_dict = {"additional_special_tokens": ["<bom>", "<eom>"]}
        self.add_special_tokens(
            special_tokens_dict, replace_additional_special_tokens=False
        )

    def mllm_setup(self, num_graph_tokens: int):
        # if self.pad_token is None:
        #     logger.warning(f"Tokenizer {self.__class__} has no pad_token. Use unk_token instead.")
        #     self.pad_token = self.unk_token

        self.num_graph_tokens = num_graph_tokens

        # Currently we only support the image modality for media modality.
        self.media_tokens = {
            k: -int(i + 1) for i, k in enumerate(MEDIA_TOKENS["graph"])
        }
        self.media_lengths = {
            MEDIA_TOKENS["graph"][0]: num_graph_tokens,
            MEDIA_TOKENS["graph"][1]: num_graph_tokens,
        }  # token lengths

    def encode_prompt(self, prompt, max_length, no_eos=False):
        """Tokenize prompt which consists of image-text or text only, with role tokens.
        Role pattern is "AI: " or "Human: ".

        Args:
            prompt
            max_length (int or None): here, max_length is used for truncation.
                If max_length is None, no truncation is applied.
            no_eos: if True, eos token is not added at the end of the prompt.
                Note that eos token is still used for end-of-AI-turn token even no_eos=True.
        """
        max_length = (
            max_length or _INFINITE
        )  # if None, set to infinite for no-truncation

        # output enc_chunk
        if self.bos_token_id is None:
            enc_chunk = []
            label_chunk = []
        else:
            enc_chunk = [self.bos_token_id]
            label_chunk = [0]

        pattern = "|".join(map(re.escape, list(self.media_tokens.keys())))

        for idx, m in enumerate(prompt):
            if len(enc_chunk) > max_length:
                break

            if m["role"] == "system":
                curr_message = m["content"] + "\n\n"
                curr_chunk = self(curr_message, add_special_tokens=False)["input_ids"]
                enc_chunk += curr_chunk
                label_chunk += [0] * len(curr_chunk)

            elif m["role"] == "user":
                curr_message = "User: " + m["content"] + "\n\n"
                chunk_strs = re.split(f"({pattern})", curr_message)
                chunk_strs = [x.strip() for x in chunk_strs if len(x) > 0]
                for _, chunk_str in enumerate(chunk_strs):
                    if chunk_str in self.media_tokens:
                        enc_chunk += [
                            self.media_tokens[chunk_str]
                        ] * self.media_lengths[chunk_str]
                        label_chunk += [0] * self.media_lengths[chunk_str]
                    else:
                        curr_chunk = self(chunk_str, add_special_tokens=False)[
                            "input_ids"
                        ]
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)

            elif m["role"] == "assistant":
                curr_message = "Assistant: " + m["content"] + "\n\n"
                curr_chunk = self(curr_message, add_special_tokens=False)["input_ids"]
                enc_chunk += curr_chunk
                label_chunk += [0] * 2 + [1] * (len(curr_chunk) - 2)
            else:
                raise ValueError("Role should be either system, user, or assistant")

        if no_eos and enc_chunk[-1] == self.eos_token_id:
            # the last token can be != eos_token_id; when the prompt is ended with `AI: `.
            # in this case, there is no AI-answer, thus, no eos token is added.
            enc_chunk = enc_chunk[:-1]
            label_chunk = label_chunk[:-1]

        enc_chunk = enc_chunk[: max_length + 1]
        label_chunk = label_chunk[: max_length + 1]
        L = len(enc_chunk)
        assert L == len(label_chunk)

        input_ids = torch.as_tensor(enc_chunk, dtype=torch.long)
        loss_mask = torch.as_tensor(label_chunk, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        # Label
        labels = input_ids.clone()
        labels[loss_mask != 1] = IGNORE_INDEX

        # The length of input_ids (L) includes <bos> and <eos> tokens.
        # Since max_length does not include <bos> token, L <= max_length + 1
        assert L <= max_length + 1

        return {
            "input_ids": input_ids,  # [L]
            "labels": labels,  # [L]
            "seq_length": L,  # int
            "attention_mask": attention_mask,  # [L]
        }

    def batch_encode_prompt(
        self,
        prompts,
        padding="longest",
        padding_side="right",
        max_length=None,
        no_eos=False,
    ):
        """Batch encode prompts, pad/truncate to the same length, and collate them.
        Args:
            prompts (list[str])
            padding ("longest" or "max_length")
            padding_side ("left" or "right")
            pad_value (int)
            max_length (int or None): if padding == "max_length", max_length should be given
        """
        batch = [self.encode_prompt(prompt, max_length, no_eos) for prompt in prompts]
        batch = self.batch_collate_pad(batch, padding, padding_side, max_length)

        return batch

    def batch_collate_pad(
        self,
        batch,
        padding,
        padding_side,
        max_length,
        is_eval=False,
    ):
        """Collate batch and pad/truncate to the same length
        Args:
            batch
            padding ("longest" or "max_length")
            padding_side ("left" or "right")
            pad_value (int)
            max_length (int or None): if padding == "max_length", max_length should be given
        """
        if padding == "max_length":
            assert max_length is not None, (
                "max_length should be given if padding == 'max_length'"
            )
        else:
            # if padding == 'longest' and max_length is None, set to infinite for no-truncation
            max_length = max_length or _INFINITE

        label_pad_value = IGNORE_INDEX
        input_padding_side = padding_side
        label_padding_side = padding_side
        if is_eval:
            label_pad_value = self.pad_token_id
            input_padding_side = "left"
            label_padding_side = "right"

        input_ids = [sample["input_ids"] for sample in batch]
        labels = [sample["labels"] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        seq_length = [sample["seq_length"] for sample in batch]

        # max_length + 1 for bos_token
        input_ids = _pad_trunc(
            input_ids, padding, input_padding_side, self.pad_token_id, max_length + 1
        )
        labels = _pad_trunc(
            labels, padding, label_padding_side, label_pad_value, max_length + 1
        )
        attention_mask = _pad_trunc(
            attention_mask, padding, input_padding_side, 0, max_length + 1
        )
        seq_length = torch.as_tensor(seq_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "seq_length": seq_length,
        }


def build_llamo_tokenizer(
    pretrained_tokenizer_name_or_path, num_graph_tokens: int, use_selfies=False
):
    """Build LLaMo tokenizer"""
    # if "Meta-Llama-3" in pretrained_tokenizer_name_or_path:
    # is_llama3 = True
    if "falcon" in pretrained_tokenizer_name_or_path:
        is_llama3 = True
    else:
        is_llama3 = False

    if is_llama3:
        tokenizer = LLaMoTokenizerforFalcon.from_pretrained(
            pretrained_tokenizer_name_or_path, use_fast=False, add_eos_token=True
        )
    else:
        tokenizer = LLaMoTokenizer.from_pretrained(
            pretrained_tokenizer_name_or_path, use_fast=False, add_eos_token=True
        )
    tokenizer.mllm_setup(num_graph_tokens)
    if use_selfies:
        tokenizer.selfies_setup()

    return tokenizer
