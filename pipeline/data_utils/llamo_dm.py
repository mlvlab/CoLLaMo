# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#### delete after debugging ####
import math
from functools import partial

import torch
from pytorch_lightning import LightningDataModule

# import torch_geometric
# from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ..collate import batchify, batchify3d, batchify_unify
from .datasets import load_dataset
from .datasets.base3d import D3Collater


class LLaMoDM(LightningDataModule):
    def __init__(
        self,
        mode: str = "pretrain",
        num_workers: int = 4,
        batch_size: int = 256,
        root_train: str = "data/PubChem324kV2/",
        root_eval: str = "data/PubChem324kV2/",
        max_length: int = 256,
        tokenizer=None,
        uni_dict=None,
        args=None,
        config=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        # self.selfies_prompt = config.selfies_prompt
        self.mtl = args.mtl
        self.balance = args.balance
        self.use_selfies = args.use_selfies
        self.tokenizer = tokenizer

        self.d3_collater = D3Collater(uni_dict.pad())

        print(self.mtl)

        if "graphs_prompt" in config.to_dict():
            self.graphs_prompt = config.graphs_prompt
        else:
            self.graphs_prompt = None

        if "use_repr" in config.to_dict():
            self.use_repr = config.use_repr
        else:
            self.use_repr = True

        if "biot5" in config.lm_config.pretrained_lm_name_or_path:
            self.biot5 = True
        else:
            self.biot5 = False

        if "encode_mode" in config.to_dict():
            self.encode_mode = config.encode_mode
            if config.encode_mode == "1d":
                self.not_use_kg = True
            else:
                self.not_use_kg = False
        else:
            self.not_use_kg = False
            self.encode_mode = "3d"

        if self.balance:
            data_name_list = [
                "new_MoleculeDesc_final",
                "new_motif_final",
                "new_naming_final",
                "new_Pubchemqc_final",
                "new_Pubchemproperty_final",
                "new_Pubchemdesc_final",
            ]
            self.dataset_list = []
            for data_name in data_name_list:
                if self.biot5:
                    self.dataset_list.append(
                        load_dataset(
                            "BaseBiot5",
                            root_train + f"{data_name}/train.pt",
                            self.tokenizer,
                            self.max_length,
                            self.not_use_kg,
                            self.use_repr,
                            self.use_selfies,
                            uni_dict,
                        )
                    )
                else:
                    self.dataset_list.append(
                        load_dataset(
                            "BaseUnify",
                            root_train + f"{data_name}/train.pt",
                            self.tokenizer,
                            self.max_length,
                            self.not_use_kg,
                            self.use_repr,
                            self.use_selfies,
                            uni_dict,
                        )
                    )

            if self.biot5:
                self.val_dataset = load_dataset(
                    "BaseBiot5",
                    root_eval + "test.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
                self.test_dataset = load_dataset(
                    "BaseBiot5",
                    root_eval + "test.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
            else:
                self.val_dataset = load_dataset(
                    "BaseUnify",
                    root_eval + "test.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
                self.test_dataset = load_dataset(
                    "BaseUnify",
                    root_eval + "test.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )

        elif self.encode_mode in ["1d", "2d"]:
            self.pretrain_dataset = load_dataset(
                "Base",
                root_train + "train.pt",
                self.tokenizer,
                self.max_length,
                self.not_use_kg,
                self.use_repr,
                self.use_selfies,
                uni_dict,
            )
            self.train_dataset = load_dataset(
                "Base",
                root_train + "train.pt",
                self.tokenizer,
                self.max_length,
                self.not_use_kg,
                self.use_repr,
                self.use_selfies,
                uni_dict,
            )
            self.val_dataset = load_dataset(
                "Base",
                root_eval + "test.pt",
                self.tokenizer,
                self.max_length,
                self.not_use_kg,
                self.use_repr,
                self.use_selfies,
                uni_dict,
            )
            self.test_dataset = load_dataset(
                "Base",
                root_eval + "test.pt",
                self.tokenizer,
                self.max_length,
                self.not_use_kg,
                self.use_repr,
                self.use_selfies,
                uni_dict,
            )
        elif self.encode_mode == "unify":
            if self.biot5:
                self.pretrain_dataset = load_dataset(
                    "BaseBiot5",
                    root_train + "train.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
                self.train_dataset = load_dataset(
                    "BaseBiot5",
                    root_train + "train.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
                self.val_dataset = load_dataset(
                    "BaseBiot5",
                    root_eval + "test.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
                self.test_dataset = load_dataset(
                    "BaseBiot5",
                    root_eval + "test.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
            else:
                self.pretrain_dataset = load_dataset(
                    "BaseUnify",
                    root_train + "train.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
                self.train_dataset = load_dataset(
                    "BaseUnify",
                    root_train + "train.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
                self.val_dataset = load_dataset(
                    "BaseUnify",
                    root_eval + "test.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
                self.test_dataset = load_dataset(
                    "BaseUnify",
                    root_eval + "test.pt",
                    self.tokenizer,
                    self.max_length,
                    self.not_use_kg,
                    self.use_repr,
                    self.use_selfies,
                    uni_dict,
                )
        else:
            self.pretrain_dataset = load_dataset(
                "Base3d",
                root_train + "train.pt",
                self.tokenizer,
                self.max_length,
                self.not_use_kg,
                self.use_repr,
                self.use_selfies,
                uni_dict,
            )
            self.train_dataset = load_dataset(
                "Base3d",
                root_train + "train.pt",
                self.tokenizer,
                self.max_length,
                self.not_use_kg,
                self.use_repr,
                self.use_selfies,
                uni_dict,
            )
            self.val_dataset = load_dataset(
                "Base3d",
                root_eval + "test.pt",
                self.tokenizer,
                self.max_length,
                self.not_use_kg,
                self.use_repr,
                self.use_selfies,
                uni_dict,
            )
            self.test_dataset = load_dataset(
                "Base3d",
                root_eval + "test.pt",
                self.tokenizer,
                self.max_length,
                self.not_use_kg,
                self.use_repr,
                self.use_selfies,
                uni_dict,
            )

    def create_size_based_dataloader(self, collate_fn):
        concat_dataset = ConcatDataset(self.dataset_list)
        sqrt_lengths = [math.sqrt(len(dataset)) for dataset in self.dataset_list]
        total_len = sum(sqrt_lengths)

        # weights = [length / total_len for length in sqrt_lengths]
        weights = []
        for i, length in enumerate(sqrt_lengths):
            weights.extend([length / total_len] * len(self.dataset_list[i]))

        assert len(weights) == len(concat_dataset)
        weights = torch.DoubleTensor(weights)
        # sampler = WeightedRandomSampler(weights, num_samples=2400000, replacement=True)
        sampler = WeightedRandomSampler(weights, num_samples=240000, replacement=True)

        loader = DataLoader(
            concat_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

        return loader

    def train_dataloader(self):
        if self.balance:
            collate_fn = partial(
                batchify_unify,
                tokenizer=self.tokenizer,
                d3_collater=self.d3_collater,
                max_length=self.max_length,
                use_trunc=True,
                is_eval=False,
            )
        elif self.encode_mode in ["1d", "2d"]:
            collate_fn = partial(
                batchify,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                use_trunc=True,
                is_eval=False,
            )
        elif self.encode_mode == "unify":
            collate_fn = partial(
                batchify_unify,
                tokenizer=self.tokenizer,
                d3_collater=self.d3_collater,
                max_length=self.max_length,
                use_trunc=True,
                is_eval=False,
            )

        else:
            collate_fn = partial(
                batchify3d,
                tokenizer=self.tokenizer,
                d3_collater=self.d3_collater,
                max_length=self.max_length,
                use_trunc=True,
                is_eval=False,
            )

        if self.mode == "pretrain":
            # sampler = DistributedSampler(self.pretrain_dataset) if not isinstance(self.pretrain_dataset, IterableDataset) else None
            if self.balance:
                loader = self.create_size_based_dataloader(collate_fn)
            else:
                loader = DataLoader(
                    self.pretrain_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=False,
                    drop_last=True,
                    persistent_workers=True,
                    collate_fn=collate_fn,
                )
        elif self.mode == "ft":
            if self.balance:
                loader = self.create_size_based_dataloader(collate_fn)
            else:
                loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=False,
                    drop_last=True,
                    persistent_workers=True,
                    collate_fn=collate_fn,
                )
        else:
            raise NotImplementedError
        return loader

    def val_dataloader(self):
        if self.encode_mode in ["1d", "2d"]:
            collate_fn = partial(
                batchify,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                use_trunc=True,
                is_eval=True,
            )
        elif self.encode_mode == "unify":
            collate_fn = partial(
                batchify_unify,
                tokenizer=self.tokenizer,
                d3_collater=self.d3_collater,
                max_length=self.max_length,
                use_trunc=True,
                is_eval=True,
            )
        else:
            collate_fn = partial(
                batchify3d,
                tokenizer=self.tokenizer,
                d3_collater=self.d3_collater,
                max_length=self.max_length,
                use_trunc=True,
                is_eval=True,
            )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        return [val_loader, test_loader]

    def test_dataloader(self):
        if self.encode_mode in ["1d", "2d"]:
            collate_fn = partial(
                batchify,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                use_trunc=True,
                is_eval=True,
            )
        elif self.encode_mode == "unify":
            collate_fn = partial(
                batchify_unify,
                tokenizer=self.tokenizer,
                d3_collater=self.d3_collater,
                max_length=self.max_length,
                use_trunc=True,
                is_eval=True,
            )
        else:
            collate_fn = partial(
                batchify3d,
                tokenizer=self.tokenizer,
                d3_collater=self.d3_collater,
                max_length=self.max_length,
                use_trunc=True,
                is_eval=True,
            )
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--inference_batch_size", type=int, default=4)
        parser.add_argument("--root_train", type=str, default="data/PubChemDataset_v4")
        parser.add_argument("--root_eval", type=str, default="data/PubChemDataset_v4")
        parser.add_argument("--filtered_cid_path", type=str, default=None)
        # parser.add_argument('--mtl', type=str, default=None)
        parser.add_argument("--mtl", action="store_true", default=False)
        parser.add_argument("--balance", action="store_true", default=False)

        return parent_parser
