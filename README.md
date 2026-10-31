<div align="center">

# CoLLaMo

### Improving Large Molecular Language Model via Relation-aware Multimodal Collaboration

Jinyoung Park · Minseong Bae · Jeehye Na · Hyunwoo J. Kim

KAIST

**AAAI 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2601.12256-b31b1b.svg)](https://arxiv.org/abs/2601.12256)
[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-0b5cad.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/37058)
[![Built on LLaMo](https://img.shields.io/badge/built%20on-LLaMo%20%28NeurIPS%202024%29-6e7781.svg)](https://github.com/mlvlab/LLaMo)

</div>

<!-- TODO(author): add paper Fig. 1 as assets/overview.png (or drag the image into the GitHub editor and use the generated URL). -->
<p align="center">
  <img src="assets/overview.png" alt="CoLLaMo architecture" width="92%">
</p>

> **TL;DR.** Large molecular language models often hallucinate because they look at a molecule through a single view, or stitch views together only loosely. **CoLLaMo** reads every molecule as a **1D SELFIES string, a 2D graph, and a 3D conformer** at once, and fuses them with a projector whose attention is guided by how atoms relate to each other, both in the graph and in space. The result is a molecular assistant that stays better grounded in the input molecule and is stronger across captioning, property QA, motif counting, and IUPAC naming.

## News

- **2026.03** · Training and inference code released.
- **2026.01** · CoLLaMo appears at **AAAI 2026**; the paper is on [arXiv](https://arxiv.org/abs/2601.12256).

## Results

Selected numbers from Tables 2–4 of the paper. See the paper for the full set of baselines and settings.

| Task | Metric | Compared to | Baseline | **CoLLaMo** |
|---|---|---|---:|---:|
| Molecule captioning | BLEU / METEOR ↑ | LLaMo | 38.9 / 67.1 | **44.5 / 70.5** |
| IUPAC name prediction | BLEU / METEOR ↑ | LLaMo | 56.0 / 73.2 | **59.8 / 77.0** |
| Motif counting | MAE ↓ | GPT-4o (ICL) | 2.45 | **1.31** |
| Descriptive property QA (generalist) | BLEU / METEOR ↑ | 3D-MoLM | 26.08 / 51.93 | **45.62 / 59.89** |
| Computed property QA (specialist) | MAE ↓ (LogP / TPSA / HOMO / SCF) | 3D-MoLM | 0.66 / 9.71 / 0.26 / 0.35 | **0.37 / 4.93 / 0.08 / 0.02** |
| Caption hallucination | CHARM ↓ | LLaMo | 64.7 | **58.5** |
| Caption quality (GPT-4o judge, 0–5) | LLM score ↑ | LLaMo | 2.17 | **2.52** |

<sub>Units: TPSA in Å², HOMO in eV, SCF energy in 10⁴ eV.</sub>

## Getting Started

### 1. Environment

```bash
git clone https://github.com/mlvlab/CoLLaMo.git
cd CoLLaMo

conda create -n collamo python=3.9 -y
conda activate collamo

# PyTorch (CUDA 11.8) and PyG
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.5.0
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

# Model, training, chemistry, and evaluation
pip install numpy==1.24.4 transformers==4.44.2 peft==0.9.0 pytorch-lightning==2.2.1 \
    salesforce-lavis==1.0.0 sentencepiece omegaconf pyyaml \
    rdkit==2023.9.5 selfies==2.1.1 ogb==1.3.5 \
    nltk rouge-score scikit-learn Levenshtein
```

### 2. Pretrained encoders and LLM

```bash
mkdir -p MoleculeSTM

# 3D conformer encoder: Uni-Mol molecular pretraining weights (without hydrogens)
wget -P MoleculeSTM https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt
```

- **2D graph encoder.** Download the [MoleculeSTM](https://github.com/chao1224/MoleculeSTM) GIN checkpoint ([Google Drive](https://drive.google.com/file/d/1oXb3BoDUZPwRiTYJSdTJRUwMLxWT8NTm/view?usp=sharing)) and save it as `MoleculeSTM/molecule_model.pth`.
- **LLM.** CoLLaMo uses [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). Accept the license on Hugging Face and run `huggingface-cli login`, or point the config to a local copy.

### 3. Data

<!-- TODO(author): replace "coming soon" with the CoLLaMo data and checkpoint links. -->

The tokenizer loads the SELFIES vocabulary from `data/`, so copy it there first:

```bash
mkdir -p data && cp selfies_alphabet.txt data/
```

### 4. Configuration

Model and optimization settings live in YAML files under `config_file/`.

## Training

`<CKPT_DIR>` below is the checkpoint and log root set in `train.py`; each run is saved to `<CKPT_DIR>/<filename>/`. Training runs in `bf16-mixed` precision by default.

**Stage 1 · Molecule–language alignment.** Trains the projector and the molecule encoders while the LLM stays frozen.

```bash
python train.py --mode pretrain \
    --config_file config_file/stage1.yaml \
    --root_train data/pretrain/ --root_eval data/pretrain/ \
    --use_selfies --devices 0,1,2,3 \
    --batch_size 4 --inference_batch_size 16 --max_epochs 3 \
    --filename collamo_stage1
```

**Stage 2 · Multi-task instruction tuning.** Trains the projector and the LLM (LoRA) while the encoders stay frozen.
```bash
python train.py --mode ft \
    --config_file config_file/stage2.yaml \
    --root_train data/instruction_tuning/ \
    --root_eval data/instruction_tuning/new_MoleculeDesc_final/ \
    --balance --use_selfies --devices 0,1,2,3 \
    --batch_size 4 --inference_batch_size 16 --max_epochs 3 \
    --filename collamo_stage2 \
    --stage_path <CKPT_DIR>/collamo_stage1/last.ckpt
```

## Inference and Evaluation

Generate predictions for a task with a trained checkpoint (molecule captioning shown):

```bash
python train.py --mode eval \
    --config_file config_file/stage2.yaml \
    --root_train data/instruction_tuning/new_MoleculeDesc_final/ \
    --root_eval data/instruction_tuning/new_MoleculeDesc_final/ \
    --use_selfies --devices 0,1,2,3 \
    --batch_size 1 --inference_batch_size 1 \
    --filename collamo_eval_desc \
    --stage_path <path/to/collamo_checkpoint.ckpt>
```

Score them with:

```bash
python evaluate.py --task desc \
    --path <CKPT_DIR>/collamo_eval_desc/lightning_logs/version_0/predictions.txt
```

| `--task` | Benchmark | Metrics |
|---|---|---|
| `desc` | Molecule captioning, descriptive property QA | BLEU-2/4, ROUGE-1/2/L, METEOR |
| `naming` | IUPAC name prediction | BLEU, ROUGE, METEOR, exact match |
| `motif` | Motif counting | MAE |
| `propqc2d` | Computed property QA: weight, LogP, TPSA, complexity | MAE |
| `propqc3d` | Computed property QA: HOMO, LUMO, HOMO-LUMO gap, SCF energy | MAE |

## Citation

If you find CoLLaMo useful, please consider citing our paper and giving the repository a ⭐.

```bibtex
@inproceedings{park2026collamo,
  title     = {Improving Large Molecular Language Model via Relation-aware Multimodal Collaboration},
  author    = {Park, Jinyoung and Bae, Minseong and Na, Jeehye and Kim, Hyunwoo J.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```
