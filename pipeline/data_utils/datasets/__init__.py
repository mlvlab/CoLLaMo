from .base import BaseDataset
from .base3d import Base3dDataset
from .base_biot5 import BaseDatasetBioT5
from .base_dpo import BaseDPODataset
from .base_unify import BaseUnifyDataset

DATASET_CLASS_DICT = {
    "Base": BaseDataset,
    "Base3d": Base3dDataset,
    "BaseUnify": BaseUnifyDataset,
    "BaseBiot5": BaseDatasetBioT5,
    "BaseDPO": BaseDPODataset,
}


def load_dataset(
    dset_name,
    input_path,
    tokenizer,
    max_length,
    not_use_kg,
    use_repr,
    use_selfies,
    uni_dict,
    **kwargs,
):
    dataset_class = DATASET_CLASS_DICT[dset_name]
    dset = dataset_class(
        input_path,
        tokenizer,
        max_length,
        not_use_kg,
        use_repr,
        use_selfies,
        uni_dict,
        **kwargs,
    )

    return dset
