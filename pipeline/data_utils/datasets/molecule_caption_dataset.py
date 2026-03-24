from collections import namedtuple

import torch

# from torch_geometric.data import Dataset, InMemoryDataset
from .base import BaseDataset

Example = namedtuple(
    "Example", ["index", "graph", "smiles", "input_prompt", "text", "data_dict"]
)


class MoleculeCaptionDataset(BaseDataset):
    def __init__(
        self,
        input_path,
        tokenizer,
        max_length,
        smiles_selfies_prompt=None,
        graphs_prompt=None,
        lm="llama",
        not_use_kg=False,
        mtl=False,
    ):
        super().__init__(tokenizer, max_length)
        self.data, self.slices = torch.load(input_path)

        self.input_path = input_path
        self.max_length = max_length
        self.lm = lm

        if not smiles_selfies_prompt:
            smiles_selfies_prompt = (
                "The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. "
            )
        if not graphs_prompt:
            graphs_prompt = "<graph>"

        self.smiles_selfies_prompt = smiles_selfies_prompt
        self.graphs_prompt = graphs_prompt
        self.not_use_kg = not_use_kg

    # def load_data(self, raw)
    def __getitem__(self, index):
        if self.perm is not None:
            index = self.perm[index]
        data = self.get(index)
        smiles = data.smiles
        instruction = "Could you provide a description of this molecule?"
        assert len(smiles.split("\n")) == 1

        input_prompt = self.build_prompt(
            smiles_selfies_prompt=self.smiles_selfies_prompt,
            graphs_prompt=self.graphs_prompt,
            lm=self.lm,
            instruction=instruction,
            not_use_kg=self.not_use_kg,
        )
        if input_prompt.find("{}") >= 0:
            input_prompt = input_prompt.format(smiles[:128])
        else:
            input_prompt = input_prompt
        text = data.text.split("\n")[:100]
        # text = ' '.join(text)
        text = " ".join(text) + "\n"
        # we explain graph, text, and input_prompt Need to check the type of each data.
        data_dict = {
            "answer": text,
            "graph_id": data.cid,
        }
        if self.not_use_kg:
            data = None
        ex = Example(index, data, smiles, input_prompt, text, data_dict)
        return ex


if __name__ == "__main__":
    dataset = MoleculeCaptionDataset("./data/PubChem324kV2/pretrain.pt", 128)
    print(dataset[0])
