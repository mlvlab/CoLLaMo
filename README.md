<p align="center">
  <h1 align="center">Improving Large Molecular Language Model via Relation-aware Multimodal Collaboration

</h1>
  
  <p align="center">Jinyoung Park, Minseong Bae, Jeehye Na, Hyunwoo J. Kim.
  </p>

</p>
Official PyTorch implementation of the "Improving Large Molecular Language Model via Relation-aware Multimodal Collaboration".
(AAAI 2026)


## Enviroment
To install requirements, run:
```bash
git clone https://github.com/mlvlab/LLaMo.git
cd LLaMo
conda create -n llamo python==3.9
conda activate llamo
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Preparation
### Pretrained graph encoder
We utilized the pre-trained graph encoder checkpoint from the [MoleculeSTM](https://github.com/chao1224/MoleculeSTM?tab=readme-ov-file) repository. 
You can download the pre-trained graph encoder checkpoint from the [link](https://drive.google.com/file/d/1oXb3BoDUZPwRiTYJSdTJRUwMLxWT8NTm/view?usp=sharing).
Place the pretrained graph model in the `MoleculeSTM/' folder.

### Datasets
You can download the datasets from the [link](https://drive.google.com/drive/folders/1Lr18nbolJnxIUbPHvn2qlwUqTouSgkeE?usp=drive_link).
Place both datasets (MoleculeDesc, instruction_tuning) in the `data/` folder.

### Checkpoint
You can download our checkpoint from the [link](https://drive.google.com/file/d/19zYlIwWY5Oemur-1Nv093B1HSuRDiLot/view?usp=sharing).

---

We're now working on refactoring the code to incorporate the huggingface.
Please stay tuned:)

## Training
You can update the training config in the `config_file` folder.
### Step1. Molecular graph-language alignment
```bash
python train.py --root_train 'data/MoleculeDesc/' --root_eval 'data/MoleculeDesc/' --devices '0,1,2,3' --filename "stage1" --max_epochs 3 --mode train --inference_batch_size 16 --batch_size 4 --config_file config_file/stage1.yaml --accumulate_grad_batches 4
```

### Step2. Instruction tuning
```bash
python train.py --root_train 'data/instruction_tuning/' --root_eval 'data/MoleculeDesc/' --devices '0,1,2,3' --filename "stage2" --max_epochs 3 --mode train --inference_batch_size 16 --batch_size 4 --config_file config_file/stage2.yaml --accumulate_grad_batches 4 --stage_path "./all_checkpoints/stage1/last.ckpt"
```

## Inference and Evaluation

### Inference
If you want to generate the output of the LLaMo on the molecule description generation task, you can run the following command.
```bash
python train.py --root_train 'data/MoleculeDesc/' --root_eval 'data/MoleculeDesc/' --devices '0,1,2,3' --filename "desc_output" --mode eval --inference_batch_size 1 --batch_size 1 --config_file config_file/stage2.yaml --stage_path <path_to_checkpoint>
```

### Evaluation
If you want to evaluate the performance of the LLaMo on the molecule description generation task, you can run the following command.
```bash
python evaluate.py --task desc --path <path_to_predictions>
```


## Contact
If you have any questions, please create an issue on this repository or contact at lpmn678@korea.ac.kr.

## Citation
If you find our work interesting, please consider giving a ⭐ and citation.
```bibtex
@inproceedings{park2024llamo,
  title={Improving Large Molecular Language Model via Relation-aware Multimodal Collaboration},
  author={Park, Jinyoung and Bae, Minseong and Na, Jeehye and Kim, Hyunwoo J},
  booktitle={AAAI},
  year={2026}
}
```
