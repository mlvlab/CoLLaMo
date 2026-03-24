import torch
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater

from pipeline.data_utils.constants import IGNORE_INDEX


def eval_func(text_batch):
    new_batch = []
    for t in text_batch:
        label_mask = t["labels"] != IGNORE_INDEX
        input_mask = t["labels"] == IGNORE_INDEX
        input_ids = t["input_ids"][input_mask]
        attention_mask = t["attention_mask"][input_mask]
        labels = t["labels"][label_mask]
        seq_length = input_ids.size(0)
        new_batch.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "seq_length": seq_length,
            }
        )
    return new_batch


def pad_to_length(tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
        return tensor[:, :length]
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value
                * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def batchify(batch, tokenizer, max_length: int, use_trunc=True, is_eval=False):
    """collate_fn
    Args:
        batch
        tokenizer
        max_length (int)
        use_trunc (bool)

    NOTE data["image"] can be None (e.g., text-instruction dataset)
    NOTE every batch for each device SHOULD have one image at least;
        if all batch data are text-only ones, error would occurs.
    """
    output_batch = {}
    graph_list = [data["graph"] for data in batch]
    smiles_list = [data["smiles"] for data in batch]

    num_graphs_per_sample = torch.LongTensor(
        [graph is not None for graph in graph_list]
    )
    # 1. remove None images from graph_list
    graph_list = [graph for graph in graph_list if graph is not None]

    # 2. collate for images: [num_images, c, h, w]
    output_batch["graph3d_values"] = None
    output_batch["graph_values"] = Collater([], [])(graph_list)
    # 3. collate for text
    text_batch = [data["text"] for data in batch]
    if is_eval:
        text_batch = eval_func(text_batch)
    padding = "longest" if use_trunc else "max_length"
    text_batch = tokenizer.batch_collate_pad(
        text_batch,
        padding=padding,
        padding_side="right",
        max_length=max_length,
        is_eval=is_eval,
    )

    # NOTE [bw-compat] Do not use attention mask for training, it will be generated automatically.
    # text_batch.pop("attention_mask")

    output_batch.update(
        {
            **text_batch,
            "num_graphs": num_graphs_per_sample,
        }
    )

    # output_batch = {k: v.to("cuda") for k, v in output_batch.items()}
    output_batch.update({"smiles": smiles_list})

    return output_batch


def batchify3d(
    batch, tokenizer, d3_collater, max_length: int, use_trunc=True, is_eval=False
):
    """collate_fn
    Args:
        batch
        tokenizer
        max_length (int)
        use_trunc (bool)

    NOTE data["image"] can be None (e.g., text-instruction dataset)
    NOTE every batch for each device SHOULD have one image at least;
        if all batch data are text-only ones, error would occurs.
    """
    output_batch = {}
    graph_list = [data["graph"] for data in batch]
    smiles_list = [data["smiles"] for data in batch]

    num_graphs_per_sample = torch.LongTensor(
        [graph is not None for graph in graph_list]
    )
    # 1. remove None images from graph_list
    graph_list = [graph for graph in graph_list if graph is not None]

    # 2. collate for images: [num_images, c, h, w]
    atom_vec_list = [graph["atom_vec"] for graph in graph_list]
    coordinates_list = [graph["coordinates"] for graph in graph_list]
    edge_type_list = [graph["edge_type"] for graph in graph_list]
    dist_list = [graph["dist"] for graph in graph_list]

    padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = (
        d3_collater(
            (atom_vec_list, coordinates_list, edge_type_list, dist_list, smiles_list)
        )
    )
    output_batch["graph3d_values"] = (padded_atom_vec, padded_dist, padded_edge_type)

    output_batch["graph_values"] = None
    # 3. collate for text
    text_batch = [data["text"] for data in batch]
    if is_eval:
        text_batch = eval_func(text_batch)
    padding = "longest" if use_trunc else "max_length"
    text_batch = tokenizer.batch_collate_pad(
        text_batch,
        padding=padding,
        padding_side="right",
        max_length=max_length,
        is_eval=is_eval,
    )

    # NOTE [bw-compat] Do not use attention mask for training, it will be generated automatically.
    # text_batch.pop("attention_mask")

    output_batch.update(
        {
            **text_batch,
            "num_graphs": num_graphs_per_sample,
        }
    )

    # output_batch = {k: v.to("cuda") for k, v in output_batch.items()}
    output_batch.update({"smiles": smiles_list})

    return output_batch


def batchify_unify(
    batch, tokenizer, d3_collater, max_length: int, use_trunc=True, is_eval=False
):
    """collate_fn
    Args:
        batch
        tokenizer
        max_length (int)
        use_trunc (bool)

    NOTE data["image"] can be None (e.g., text-instruction dataset)
    NOTE every batch for each device SHOULD have one image at least;
        if all batch data are text-only ones, error would occurs.
    """
    output_batch = {}
    graph_list = [data["graph"] for data in batch]
    return_not_use_3d_list = [data["return_not_use_3d"] for data in batch]
    smiles_list = [data["smiles"]["input_ids"][1:-1] for data in batch]
    max_smiles_len = max([len(smiles) for smiles in smiles_list])
    batch_smiles_ids = []
    for sm in smiles_list:
        smiles_input_ids = sm + [tokenizer.pad_token_id] * (max_smiles_len - len(sm))
        smiles_mask = [0] * len(sm) + [1] * (max_smiles_len - len(sm))
        batch_smiles_ids.append((smiles_input_ids, smiles_mask))

    num_graphs_per_sample = torch.LongTensor(
        [graph is not None for graph in graph_list]
    )
    # 1. remove None images from graph_list
    graph_list = [graph for graph in graph_list if graph is not None]

    # 2. collate for images: [num_images, c, h, w]
    # 2. collate for images: [num_images, c, h, w]
    atom_vec_list = [graph["atom_vec"] for graph in graph_list]
    coordinates_list = [graph["coordinates"] for graph in graph_list]
    edge_type_list = [graph["edge_type"] for graph in graph_list]
    dist_list = [graph["dist"] for graph in graph_list]

    padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = (
        d3_collater(
            (atom_vec_list, coordinates_list, edge_type_list, dist_list, smiles_list)
        )
    )
    output_batch["graph3d_values"] = (padded_atom_vec, padded_dist, padded_edge_type)

    graph_list = [Data(g["x"], g["edge_index"], g["edge_attr"]) for g in graph_list]
    output_batch["graph_values"] = Collater([], [])(graph_list)
    # 3. collate for text
    text_batch = [data["text"] for data in batch]
    if is_eval:
        text_batch = eval_func(text_batch)
    padding = "longest" if use_trunc else "max_length"
    text_batch = tokenizer.batch_collate_pad(
        text_batch,
        padding=padding,
        padding_side="right",
        max_length=max_length,
        is_eval=is_eval,
    )

    # NOTE [bw-compat] Do not use attention mask for training, it will be generated automatically.
    # text_batch.pop("attention_mask")

    output_batch.update(
        {
            **text_batch,
            "num_graphs": num_graphs_per_sample,
        }
    )

    # output_batch = {k: v.to("cuda") for k, v in output_batch.items()}
    output_batch.update({"smiles": batch_smiles_ids})
    output_batch.update({"return_not_use_3d": return_not_use_3d_list})

    return output_batch


def batchify_dpo(
    batch, tokenizer, d3_collater, max_length: int, use_trunc=True, is_eval=False
):
    """collate_fn
    Args:
        batch
        tokenizer
        max_length (int)
        use_trunc (bool)

    NOTE data["image"] can be None (e.g., text-instruction dataset)
    NOTE every batch for each device SHOULD have one image at least;
        if all batch data are text-only ones, error would occurs.
    """
    output_batch = {}
    graph_list = [data["graph"] for data in batch] + [data["graph"] for data in batch]
    return_not_use_3d_list = [data["return_not_use_3d"] for data in batch] + [
        data["return_not_use_3d"] for data in batch
    ]
    smiles_list = [data["smiles"]["input_ids"][1:-1] for data in batch] + [
        data["smiles"]["input_ids"][1:-1] for data in batch
    ]
    max_smiles_len = max([len(smiles) for smiles in smiles_list])
    batch_smiles_ids = []
    for sm in smiles_list:
        smiles_input_ids = sm + [tokenizer.pad_token_id] * (max_smiles_len - len(sm))
        smiles_mask = [0] * len(sm) + [1] * (max_smiles_len - len(sm))
        batch_smiles_ids.append((smiles_input_ids, smiles_mask))

    num_graphs_per_sample = torch.LongTensor(
        [graph is not None for graph in graph_list]
    )
    # 1. remove None images from graph_list
    graph_list = [graph for graph in graph_list if graph is not None]

    # 2. collate for images: [num_images, c, h, w]
    # 2. collate for images: [num_images, c, h, w]
    atom_vec_list = [graph["atom_vec"] for graph in graph_list]
    coordinates_list = [graph["coordinates"] for graph in graph_list]
    edge_type_list = [graph["edge_type"] for graph in graph_list]
    dist_list = [graph["dist"] for graph in graph_list]

    padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles = (
        d3_collater(
            (atom_vec_list, coordinates_list, edge_type_list, dist_list, smiles_list)
        )
    )
    output_batch["graph3d_values"] = (padded_atom_vec, padded_dist, padded_edge_type)

    graph_list = [Data(g["x"], g["edge_index"], g["edge_attr"]) for g in graph_list]
    output_batch["graph_values"] = Collater([], [])(graph_list)
    # 3. collate for text
    prompt_text_batch = [data["prompt_text_input"] for data in batch]
    chosen_text_batch = [data["chosen_text_input"] for data in batch]
    rejected_text_batch = [data["rejected_text_input"] for data in batch]
    # test_text_batch = [data["test_text_input"] for data in batch]

    padding = "longest" if use_trunc else "max_length"

    prompt_text_batch = tokenizer.batch_collate_pad(
        prompt_text_batch,
        padding=padding,
        padding_side="left",
        max_length=max_length,
        is_eval=is_eval,
    )

    chosen_text_batch = tokenizer.batch_collate_pad(
        chosen_text_batch,
        padding=padding,
        padding_side="right",
        max_length=max_length,
        is_eval=is_eval,
    )

    rejected_text_batch = tokenizer.batch_collate_pad(
        rejected_text_batch,
        padding=padding,
        padding_side="right",
        max_length=max_length,
        is_eval=is_eval,
    )

    # test_text_batch = tokenizer.batch_collate_pad(
    #     test_text_batch,
    #     padding=padding,
    #     padding_side="right",
    #     max_length=max_length,
    #     is_eval=is_eval,
    # )

    max_completion_length = min(
        max(
            chosen_text_batch["input_ids"].size(1),
            rejected_text_batch["input_ids"].size(1),
        ),
        768,
    )
    prompt_input_ids = torch.cat(
        [prompt_text_batch["input_ids"], prompt_text_batch["input_ids"]], dim=0
    )
    prompt_attention_mask = torch.cat(
        [prompt_text_batch["attention_mask"], prompt_text_batch["attention_mask"]],
        dim=0,
    )
    prompt_labels = torch.cat(
        [prompt_text_batch["labels"], prompt_text_batch["labels"]], dim=0
    )
    completion_input_ids = torch.cat(
        [
            pad_to_length(
                chosen_text_batch["input_ids"],
                max_completion_length,
                pad_value=tokenizer.pad_token_id,
            ),
            pad_to_length(
                rejected_text_batch["input_ids"],
                max_completion_length,
                pad_value=tokenizer.pad_token_id,
            ),
        ],
        dim=0,
    )
    completion_attention_mask = torch.cat(
        [
            pad_to_length(
                chosen_text_batch["attention_mask"], max_completion_length, pad_value=0
            ),
            pad_to_length(
                rejected_text_batch["attention_mask"],
                max_completion_length,
                pad_value=0,
            ),
        ],
        dim=0,
    )
    completion_labels = torch.cat(
        [
            pad_to_length(
                chosen_text_batch["labels"],
                max_completion_length,
                pad_value=IGNORE_INDEX,
            ),
            pad_to_length(
                rejected_text_batch["labels"],
                max_completion_length,
                pad_value=IGNORE_INDEX,
            ),
        ],
        dim=0,
    )

    # NOTE [bw-compat] Do not use attention mask for training, it will be generated automatically.
    # text_batch.pop("attention_mask")

    output_batch.update(
        {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_labels": prompt_labels,
            "completion_input_ids": completion_input_ids,
            "completion_attention_mask": completion_attention_mask,
            "completion_labels": completion_labels,
            "num_graphs": num_graphs_per_sample,
        }
    )

    # output_batch = {k: v.to("cuda") for k, v in output_batch.items()}
    output_batch.update({"smiles": batch_smiles_ids})
    output_batch.update({"return_not_use_3d": return_not_use_3d_list})

    return output_batch
