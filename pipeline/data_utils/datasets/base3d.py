

# from torch.utils.data import Dataset
import random

import numpy as np
import selfies as sf
import torch
from scipy.spatial import distance_matrix

from pipeline import unicore_data_utils
from pipeline.data_utils.constants import MEDIA_TOKENS, SYSTEM_MESSAGE

from .base import BaseDataset


class Base3dDataset(BaseDataset):
    """Base dataset class"""

    def __init__(
        self,
        input_path,
        tokenizer,
        max_length,
        not_use_kg,
        use_repr,
        use_selfies,
        uni_dict,
        **kwargs,
    ):
        super(Base3dDataset, self).__init__(
            input_path,
            tokenizer,
            max_length,
            not_use_kg,
            use_repr,
            use_selfies,
            uni_dict,
            **kwargs,
        )

        ## the folllowing is the default setting of uni-mol's pretrained weuights

        self.max_atoms = 256
        self.uni_dict = uni_dict

        self.bos = uni_dict.bos()
        self.eos = uni_dict.eos()
        self.num_types = len(uni_dict)

        self.remove_hydrogen = True
        self.remove_polar_hydrogen = True
        self.normalize_coords = True
        self.add_special_token = True
        self.__max_atoms = 512

    def preprocess_data(self, data):
        atoms_orig = np.array(data.atom)
        atoms = atoms_orig.copy()
        coordinate_set = data.position
        if isinstance(coordinate_set, list):
            coordinates = random.sample(coordinate_set, 1)[0].astype(np.float32)
        elif isinstance(coordinate_set, torch.Tensor):
            coordinates = coordinate_set.numpy().astype(np.float32)

        if data.atom.count("C") == 2 and len(data.atom) == 2:
            coordinates = np.random.rand(2, 3).astype(np.float32)

        assert len(atoms) == len(coordinates)
        assert coordinates.shape[1] == 3

        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            if sum(mask_hydrogen) > 0:
                atoms = atoms[mask_hydrogen]
                coordinates = coordinates[mask_hydrogen]

        if len(atoms) > self.max_atoms:
            index = np.random.permutation(len(atoms))[: self.max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]

        atom_vec = torch.from_numpy(self.uni_dict.vec_index(atoms)).long()

        if self.normalize_coords:
            coordinates = coordinates - coordinates.mean(0)

        if self.add_special_token:
            atom_vec = torch.cat(
                [torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])]
            )
            coordinates = np.concatenate(
                [np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0
            )

        edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)
        data["atom_vec"] = atom_vec
        data["edge_type"] = edge_type
        data["dist"] = dist
        data["coordinates"] = coordinates

        return data

    def build_text_from_data(self, data):
        """Build instruction text from data
        Args:
            data: datapoint given from self.data
        """
        temp_text = list()

        instruction = data.instruction
        if data.atom.count("C") == 2 and len(data.atom) == 2:
            not_use_kg = True
        else:
            not_use_kg = self.not_use_kg

        if not self.smiles_selfies_prompt:
            if self.use_selfies:
                smiles_selfies_input = "<bom>" + data.selfies + "<eom>"
                if sf.len_selfies(data.selfies) > 128:
                    splitted_selfies = list(sf.split_selfies(data.selfies))[:128]
                    smiles_selfies_input = "<bom>" + "".join(splitted_selfies) + "<eom>"
                # smiles_selfies_input = data.selfies
                # if sf.len_selfies(data.selfies) > 96:
                # splitted_selfies = list(sf.split_selfies(data.selfies))[:96]
                # smiles_selfies_input = "".join(splitted_selfies)
            else:
                smiles_selfies_input = data.smiles[:128]
        else:
            if self.use_selfies:
                smiles_selfies_input = self.smiles_selfies_prompt.format(data.selfies)
            else:
                smiles_selfies_input = self.smiles_selfies_prompt.format(data.smiles)

        graph_input = ""
        if not not_use_kg:
            graph_input = MEDIA_TOKENS["graph"][0]

        temp_text.append({"role": "system", "content": SYSTEM_MESSAGE})

        if isinstance(instruction, list):
            # limit the length of smiles_selfies_input to 128
            if self.use_repr:
                if graph_input not in instruction[0]["content"] and not not_use_kg:
                    instruction[0]["content"] = (
                        smiles_selfies_input
                        + graph_input
                        + " "
                        + instruction[0]["content"]
                    )
                else:
                    instruction[0]["content"] = (
                        smiles_selfies_input + " " + instruction[0]["content"]
                    )
            else:
                if graph_input not in instruction[0]["content"] and not not_use_kg:
                    instruction[0]["content"] = (
                        graph_input + " " + instruction[0]["content"]
                    )
                else:
                    instruction[0]["content"] = " " + instruction[0]["content"]

            temp_text.extend(instruction)
            temp_text.append({"role": "assistant", "content": data.output.strip()})

        elif isinstance(instruction, str):
            if self.use_repr:
                if graph_input not in instruction and not not_use_kg:
                    aug_instruction = (
                        smiles_selfies_input + graph_input + " " + instruction
                    )
                else:
                    aug_instruction = smiles_selfies_input + " " + instruction
            else:
                if graph_input not in instruction and not not_use_kg:
                    aug_instruction = graph_input + " " + instruction
                else:
                    aug_instruction = instruction

            temp_text.append({"role": "user", "content": aug_instruction})
            temp_text.append({"role": "assistant", "content": data.output.strip()})
        else:
            raise ValueError("Instruction should be either list or string")

        return temp_text

    def process_data(self, data):
        """Process data for the given data if required
        Args:
            data: datapoint given from self.data
        """
        data = self.preprocess_data(data)

        text = self.build_text_from_data(data)
        text_input = self.tokenizer.encode_prompt(text, self.max_length)

        if self.use_selfies:
            return {
                "graph": data,
                "raw_text": text,
                "text": text_input,
                "smiles": data.selfies,
                # "task_type": data["task_type"]
            }

        else:
            return {
                "graph": data,
                "raw_text": text,
                "text": text_input,
                "smiles": data.smiles,
                # "task_type": data["task_type"]
            }

    def __getitem__(self, index):
        data = self.get(index)
        data = self.process_data(data)

        return data


def collate_tokens_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
    return res


class D3Collater:
    def __init__(self, pad_idx, pad_to_multiple=8):
        self.pad_idx = pad_idx
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, samples):
        atom_vec, coordinates, edge_type, dist, smiles = samples
        padded_atom_vec = unicore_data_utils.collate_tokens(
            atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple
        )  # shape = [batch_size, max_atoms]
        padded_coordinates = collate_tokens_coords(
            coordinates, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple
        )  # shape = [batch_size, max_atoms, 3]
        padded_edge_type = unicore_data_utils.collate_tokens_2d(
            edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple
        )  # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = unicore_data_utils.collate_tokens_2d(
            dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple
        )  # shape = [batch_size, max_atoms, max_atoms]
        return (
            padded_atom_vec,
            padded_coordinates,
            padded_edge_type,
            padded_dist,
            smiles,
        )
