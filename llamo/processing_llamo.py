import re

import torch
import torch.utils.checkpoint
from rdkit import Chem
from torch_geometric.loader.dataloader import Collater
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

# we split individual characters inside special tokens like [START_DNA]
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", rf"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def smiles_handler(text, mol_ph, is_gal=True, graph_only=False, reverse_order=False):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)

    if graph_only:
        text = CUSTOM_SEQ_RE.sub(r"%s" % (mol_ph), text)
        return text, smiles_list
    if is_gal:
        if reverse_order:
            text = CUSTOM_SEQ_RE.sub(r"%s\1\3\4" % (mol_ph), text)
            text = escape_custom_split_sequence(text)
        else:
            text = CUSTOM_SEQ_RE.sub(r"\1\3\4%s" % (mol_ph), text)
            text = escape_custom_split_sequence(text)
        return text, smiles_list
    else:
        text = CUSTOM_SEQ_RE.sub(r"\3%s" % (mol_ph), text)
        return text, smiles_list


def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)


# Generate graph tokens for feeding to the LLM
class HoneybeeProcessor(ProcessorMixin):
    attributes = []

    def __init__(
        self,
        tokenizer,
        text_max_len,
        is_gal,
        graph_only,
        reverse_order,
        num_graph_tokens=65,
        graph_token="<graph>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokens_to_generate = 0
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.is_gal = is_gal
        self.graph_only = graph_only
        self.reverse_order = reverse_order
        self.add_BOS = True
        self.mol_ph = graph_token
        self.media_info = {self.mol_ph: num_graph_tokens}
        self.graph_collater = Collater([], [])

    def __call__(
        self,
        texts=None,
        graphs=None,
        answers=None,
        smiles=None,
        return_tensors="pt",
        **kwargs,
    ):
        if all(graph is None for graph in graphs):  # for the text only case
            graphs = None

        if texts is None and graphs is None:
            raise ValueError(
                "You have to specify either text or graphs. Both cannot be none."
            )

        if smiles is not None:
            smiles = [Chem.MolFromSmiles(smile) for smile in smiles]

        if texts is not None:
            texts = [
                smiles_handler(p, "", self.is_gal, self.graph_only, self.reverse_order)[
                    0
                ]
                for p in texts
            ]
            self.tokenizer.padding_side = "left"
            encoding = tokenize_prompts(
                prompts=texts,
                tokens_to_generate=self.tokens_to_generate,
                add_BOS=self.add_BOS,
                tokenizer=self.tokenizer,
                ignore_dist=True,
                media_info=self.media_info,
                **kwargs,
            )

        if answers is not None:
            self.tokenizer.padding_side = "right"
            answer_tokens = self.tokenizer(
                text=answers,
                truncation=True,
                padding="longest",
                add_special_tokens=True,
                max_length=self.text_max_len,
                return_tensors="pt",
                return_attention_mask=True,
            )
            encoding["answer_input_ids"] = answer_tokens["input_ids"]
            encoding["answer_attention_mask"] = answer_tokens["attention_mask"]
        else:
            encoding["answer_tokens"] = None
            encoding["answer_attention_mask"] = None

        if texts is not None and graphs is not None:
            for g in graphs:
                if "iupacs" in g.keys():
                    del g["iupacs"]
                # if 'iupac' in g.keys():
                # del g['iupac']
                if "smiles" in g.keys():
                    del g["smiles"]

            encoding["graph_values"] = self.graph_collater(graphs)
            encoding["smiles"] = smiles
            # encoding["is_null_image"] = torch.tensor(is_null_image)
            return BatchEncoding(data=encoding)
        elif texts is not None:
            encoding["graph_values"] = None
            encoding["smiles"] = smiles
            return BatchEncoding(data=encoding)
        else:
            # TODO have to checkit
            return BatchEncoding(
                data={"graph_values": graphs}, tensor_type=return_tensors
            )

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(
            *args, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(
            *args, skip_special_tokens=skip_special_tokens, **kwargs
        )


# class HoneybeeGraphProcessor(ProcessorMixin):
# pass


def detokenize_generations(
    tokens_gpu_tensor, lengths_gpu_tensor, return_segments, tokenizer
):
    """Detokenize the generated tokens."""

    prompts_plus_generations = []
    if return_segments:
        prompts_plus_generations_segments = []

    tokens = tokens_gpu_tensor.cpu().numpy().tolist()
    lengths = lengths_gpu_tensor.cpu().numpy().tolist()
    for sequence_tokens, length in zip(tokens, lengths):
        sequence_tokens = sequence_tokens[:length]
        prompts_plus_generations.append(tokenizer.detokenize(sequence_tokens))
        if return_segments:
            from tokenizers.decoders import Metaspace

            if hasattr(tokenizer, "tokenizer"):
                if isinstance(tokenizer.tokenizer.decoder, Metaspace):
                    words = tokenizer.tokenizer.decode(sequence_tokens)
                else:
                    words = []
                    for token in sequence_tokens:
                        word = tokenizer.tokenizer.decoder[token]
                        word = bytearray(
                            [tokenizer.tokenizer.byte_decoder[c] for c in word]
                        ).decode("utf-8", errors="replace")
                        words.append(word)
                prompts_plus_generations_segments.append(words)
            else:
                words = tokenizer.detokenize(sequence_tokens)
                prompts_plus_generations_segments.append(words)

    if return_segments:
        return tokens, prompts_plus_generations, prompts_plus_generations_segments

    return tokens, prompts_plus_generations


def tokenize_prompts(
    prompts=None,
    tokens_to_generate=None,
    add_BOS=None,
    rank=0,
    tokenizer=None,
    ignore_dist=False,
    **kwargs,
):
    """Tokenize prompts and make them avaiable on all ranks."""

    # On all ranks set to None so we can pass them to functions
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None

    # On the specified rank, build the above.
    attention_mask = None
    if ignore_dist or torch.distributed.get_rank() == rank:
        assert prompts is not None
        assert tokens_to_generate is not None
        # Tensor of tokens padded and their unpadded length.
        (
            prompts_tokens_cuda_long_tensor,
            prompts_length_cuda_long_tensor,
            attention_mask,
        ) = _tokenize_prompts_and_batch(
            prompts, tokens_to_generate, add_BOS, tokenizer, **kwargs
        )

    return {
        "input_ids": prompts_tokens_cuda_long_tensor,
        "attention_mask": attention_mask,
    }


def _tokenize_prompts_and_batch(
    prompts, tokens_to_generate, add_BOS, tokenizer, **kwargs
):
    """Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them into a 2D tensor.
    """

    # Tokenize all the prompts.
    prompts_tokens = [
        _tokenize_prompt(prompt, tokenizer, add_BOS, **kwargs) for prompt in prompts
    ]

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.
    prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
    # Get the max prompts length.
    max_prompt_len = max(prompts_length)
    # Number of tokens in the each sample of the batch.
    samples_length = max_prompt_len + tokens_to_generate
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
        padding_size = samples_length - prompt_length
        # extend left (ref: https://stackoverflow.com/a/19849920)
        prompt_tokens[0:0] = [tokenizer.pad_token_id] * padding_size  # left padding

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.LongTensor(prompts_tokens)
    prompts_length_tensor = torch.LongTensor(prompts_length)
    attention_mask = prompts_tokens_tensor != tokenizer.pad_token_id
    return prompts_tokens_tensor, prompts_length_tensor, attention_mask


def _tokenize_prompt(
    prompt, tokenizer, add_BOS=False, media_info={"<graph>": 65}, **kwargs
):
    media_tokens = {k: -int(i + 1) for i, k in enumerate(media_info.keys())}
    media_lengths = media_info.copy()

    if add_BOS:
        prompt_chunk = [tokenizer.bos_token_id]
    else:
        prompt_chunk = []

    # Pure Text
    if all(media_token not in prompt for media_token in media_tokens.keys()):
        enc_chunk = (
            prompt_chunk
            + tokenizer(prompt, add_special_tokens=False, **kwargs)["input_ids"]
        )
    # Multi-Modal Text
    else:
        enc_chunk = prompt_chunk
        pattern = "|".join(map(re.escape, list(media_tokens.keys())))
        chunk_strs = re.split(f"({pattern})", prompt)
        chunk_strs = [x for x in chunk_strs if len(x) > 0]
        for _, chunk_str in enumerate(chunk_strs):
            if chunk_str in media_tokens:
                enc_chunk += [media_tokens[chunk_str]] * media_lengths[chunk_str]
            else:
                tmp_chunk = tokenizer(chunk_str, add_special_tokens=False)["input_ids"]
                enc_chunk += tmp_chunk
    return enc_chunk
