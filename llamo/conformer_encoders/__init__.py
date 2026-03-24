import torch
import torch.nn as nn

from pipeline.dictionary import Dictionary

from .unimol import SimpleUniMolModel


def build_encoder(args):
    dictionary = Dictionary.load("./llamo/conformer_encoders/unimol_dict.txt")
    dictionary.add_symbol("[MASK]", is_special=True)
    unimol_model = SimpleUniMolModel(args, dictionary)
    ckpt = torch.load(
        "./MoleculeSTM/mol_pre_no_h_220816.pt", map_location=torch.device("cpu")
    )["model"]
    missing_keys, unexpected_keys = unimol_model.load_state_dict(ckpt, strict=False)

    ln_graph = nn.LayerNorm(unimol_model.num_features)

    return unimol_model, ln_graph, dictionary
