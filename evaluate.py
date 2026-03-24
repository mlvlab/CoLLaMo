import json
import os
import re

import nltk
import numpy as np
import selfies as sf
from Levenshtein import distance as lev
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rouge_score import rouge_scorer
from sklearn import metrics
from tqdm import tqdm
from transformers import LlamaTokenizer

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("omw-1.4")


def evaluate_caption(predictions, targets, tokenizer, text_trunc_length=128):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(
            gt, truncation=True, max_length=text_trunc_length
        )
        out_tokens = tokenizer.tokenize(
            out, truncation=True, max_length=text_trunc_length
        )

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        try:
            mscore = meteor_score([gt_tokens], out_tokens)
            meteor_scores.append(mscore)
        except Exception:
            continue

    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    bleu2 *= 100
    bleu4 *= 100

    print("BLEU-2 score:", bleu2)
    print("BLEU-4 score:", bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print("Average Meteor score:", _meteor_score)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print("ROUGE score:")
    rouge_1 = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores]) * 100
    print("rouge1:", rouge_1)
    print("rouge2:", rouge_2)
    print("rougeL:", rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score


def evaluate_desc(data, tokenizer, path, task, text_trunc_length=512):
    pred = [line["prediction"].strip() for line in data]
    target = [line["target"].strip() for line in data]

    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = evaluate_caption(
        pred, target, tokenizer, text_trunc_length
    )

    return {
        "Description BLEU-2": bleu2,
        "Description BLEU-4": bleu4,
        "Description ROUGE-1": rouge_1,
        "Description ROUGE-2": rouge_2,
        "Description ROUGE-L": rouge_l,
        "Description Meteor": meteor_score,
    }


def evaluate_propdesc(data, tokenizer, path, task, text_trunc_length=512):
    pred = [line["prediction"].strip() for line in data]
    target = [line["target"].strip() for line in data]

    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = evaluate_caption(
        pred, target, tokenizer, text_trunc_length
    )
    import pdb

    pdb.set_trace()
    return {
        "Property Description BLEU-2": bleu2,
        "Property Description BLEU-4": bleu4,
        "Property Description ROUGE-1": rouge_1,
        "Property Description ROUGE-2": rouge_2,
        "Property Description ROUGE-L": rouge_l,
        "Property Description Meteor": meteor_score,
    }


def evaluate_naming(data, tokenizer, path, task, text_trunc_length=512):
    pred = [line["prediction"].strip() for line in data]
    target = [line["target"].strip() for line in data]

    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = evaluate_caption(
        pred, target, tokenizer, text_trunc_length
    )
    em_list = []
    for pred, target in zip(pred, target):
        if pred.lower().strip() == target.lower().strip():
            em_list.append(1)
        else:
            em_list.append(0)
    em = np.mean(em_list)
    return {
        "Naming BLEU-2": bleu2,
        "Naming BLEU-4": bleu4,
        "Naming ROUGE-1": rouge_1,
        "Naming ROUGE-2": rouge_2,
        "Naming ROUGE-L": rouge_l,
        "Naming Meteor": meteor_score,
        "Exact Match": em,
    }


def evaluate_reaction(data, path, task):
    pred_list, test_list = [], []
    invalid_count = 0
    for d in data:
        curr_pred = sf.decoder(d["prediction"].replace(" ", "").strip())
        pred_mol = Chem.MolFromSmiles(curr_pred)
        # try:
        # curr_tgt = sf.decoder(d['target'].replace(" ", "").strip())
        # except:
        # continue
        curr_tgt = d["target"].replace(" ", "").strip()
        tgt_mol = Chem.MolFromSmiles(curr_tgt)

        canonical_pred = (
            Chem.MolToSmiles(pred_mol, isomericSmiles=False, canonical=True)
            if pred_mol
            else None
        )
        canonical_tgt = (
            Chem.MolToSmiles(tgt_mol, isomericSmiles=False, canonical=True)
            if tgt_mol
            else None
        )

        if canonical_tgt is None:
            continue

        if canonical_pred is None:
            invalid_count += 1
            continue
        pred_list.append(canonical_pred)
        test_list.append(canonical_tgt)

    references_list = []
    hypotheses_list = []
    outputs_rdkit_mols = []
    levs = []
    num_exact = 0

    for pred, test in zip(pred_list, test_list):
        pred_tokens = [c for c in pred]
        test_tokens = [c for c in test]
        references_list.append([test_tokens])
        hypotheses_list.append(pred_tokens)

        try:
            m_gt = Chem.MolFromSmiles(test)
        except Exception:
            continue
        try:
            m_out = Chem.MolFromSmiles(pred)
            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                num_exact += 1
            outputs_rdkit_mols.append((m_gt, m_out))
        except Exception:
            invalid_count += 1
            continue
        levs.append(lev(pred, test))

    blue_score = corpus_bleu(references_list, hypotheses_list)

    # Calculate similarities
    MACCS_sims, morgan_sims, RDK_sims = [], [], []
    morgan_r = 2

    for gt_m, ot_m in outputs_rdkit_mols:
        MACCS_sims.append(
            DataStructs.FingerprintSimilarity(
                MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m)
            )
        )
        RDK_sims.append(
            DataStructs.FingerprintSimilarity(
                Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m)
            )
        )
        morgan_sims.append(
            DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprint(gt_m, morgan_r),
                AllChem.GetMorganFingerprint(ot_m, morgan_r),
            )
        )

    results = {
        "BLEU": blue_score,
        "Levenshtein": sum(levs) / len(levs),
        "Exact Match": num_exact / len(test_list),
        "MACCS Similarity": np.mean(MACCS_sims),
        "RDK Similarity": np.mean(RDK_sims),
        "Morgan Similarity": np.mean(morgan_sims),
        "Invalid Count": invalid_count,
    }

    return results


def evaluate_property(data, path, task):
    pred_list = []
    tgt_list = []
    for d in data:
        try:
            pred_list.append(float(d["prediction"]))
            tgt_list.append(float(d["target"]))
        except Exception:
            continue
    import pdb

    pdb.set_trace()
    mae = metrics.mean_absolute_error(tgt_list, pred_list)

    return mae


def evaluate_propqc2d(data, path, task):

    pred_dict = {
        "Weight": [],
        "LogP": [],
        "Topological Polar Surface Area": [],
        "Complexity": [],
    }
    tgt_dict = {
        "Weight": [],
        "LogP": [],
        "Topological Polar Surface Area": [],
        "Complexity": [],
    }
    invalid_count = 0
    for d in data:
        if "Weight" in d["target"]:
            key = "Weight"
        elif "LogP" in d["target"]:
            key = "LogP"
        elif "Topological Polar Surface Area" in d["target"]:
            key = "Topological Polar Surface Area"
        elif "Complexity" in d["target"]:
            key = "Complexity"
        try:
            pattern = r"[-+]?\d*\.?\d+"
            match_pred = re.search(pattern, d["prediction"])
            match_tgt = re.search(pattern, d["target"])
            # if not match_pred or not match_tgt:
            # continue
            curr_pred = match_pred.group(0)
            curr_tgt = match_tgt.group(0)
        except Exception:
            invalid_count += 1
            continue

        if float(curr_pred) == np.inf:
            invalid_count += 1
            continue
        pred_dict[key].append(float(curr_pred))
        tgt_dict[key].append(float(curr_tgt))

    mae_dict = {}
    for key in pred_dict.keys():
        mae = metrics.mean_absolute_error(tgt_dict[key], pred_dict[key])
        mae_dict[key] = mae
        print(invalid_count)
        # print(f"{key}: {mae:.4f}")
    return mae_dict


def evaluate_propqc3d(data, path, task):
    pred_dict = {"HOMO": [], "LUMO": [], "Gap": [], "SCF": []}
    tgt_dict = {"HOMO": [], "LUMO": [], "Gap": [], "SCF": []}
    for d in data:
        if "Gap" in d["target"]:
            key = "Gap"
            temp = "The HOMO-LUMO Gap for the input molecule is"
        elif "HOMO" in d["target"]:
            key = "HOMO"
            temp = "The HOMO for the input molecule is"
        elif "LUMO" in d["target"]:
            key = "LUMO"
            temp = "The LUMO for the input molecule is"
        elif "SCF" in d["target"]:
            key = "SCF"
            temp = "The SCF Energy for the input molecule is"
        else:
            raise ValueError(f"Unknown key: {key}")
        try:
            # if key == "SCF":
            #     pattern = r"[-+]?\d*\.?\d+e\d+ eV."
            # else:
            #     pattern = r"[-+]?\d*\.?\d+ eV."
            match_pred = d["prediction"].split(temp)[1].strip()
            match_tgt = d["target"].split(temp)[1].strip()
            # if not match_pred or not match_tgt:
            # continue
            curr_pred = match_pred[:-3].strip()
            curr_tgt = match_tgt[:-3].strip()
            pred_dict[key].append(float(curr_pred))
            tgt_dict[key].append(float(curr_tgt))

        except Exception:
            # import pdb; pdb.set_trace()
            continue

        # if key == "SCF" and len(pred_dict[key]) == 0:
        # import pdb; pdb.set_trace()

    mae_dict = {}
    for key in pred_dict.keys():
        import pdb

        pdb.set_trace()
        mae = metrics.mean_absolute_error(tgt_dict[key], pred_dict[key])
        if key == "SCF":
            mae = mae / 10000
        mae_dict[key] = mae
        # print(f"{key}: {mae:.4f}")
    return mae_dict


def evaluate_motif(data, path, task):
    pred_list = []
    tgt_list = []
    for d in data:
        try:
            pred_list.append(
                float(
                    d["prediction"]
                    .split("The number of motifs in this molecule is")[1]
                    .strip()
                )
            )
            tgt_list.append(
                float(
                    d["target"]
                    .split("The number of motifs in this molecule is")[1]
                    .strip()
                )
            )
        except Exception:
            continue

    mae = metrics.mean_absolute_error(tgt_list, pred_list)
    return {"Motif": mae}


def evaluate_task(task, file=None, tokenizer=None):
    """Main function to evaluate a specific task

    Args:
        task (str): Task to evaluate ('desc', 'forward', 'retro', 'property', 'naming')
        file (str): File path to the predictions
        tokenizer: Tokenizer to use for text-based tasks
    """
    if file is None:
        file = "/all_checkpoints/temp/lightning_logs/version_0/predictions.txt"
    with open(file, "r") as f:
        data = [json.loads(line.strip()) for line in f.readlines()]

    if task in ["desc"]:
        if tokenizer is None:
            raise ValueError("Tokenizer required for description and naming tasks")
        return evaluate_desc(data, tokenizer, file, task)

    elif task in ["naming"]:
        if tokenizer is None:
            raise ValueError("Tokenizer required for description and naming tasks")
        return evaluate_naming(data, tokenizer, file, task)

    elif task in ["propdesc"]:
        if tokenizer is None:
            raise ValueError("Tokenizer required for description and naming tasks")
        return evaluate_propdesc(data, tokenizer, file, task)

    elif task in ["forward", "retro"]:
        return evaluate_reaction(data, file, task)

    elif task == "propqc2d":
        return evaluate_propqc2d(data, file, task)
    elif task == "propqc3d":
        return evaluate_propqc3d(data, file, task)

    elif "prop" in task:
        return evaluate_property(data, file, task)

    elif "motif" in task:
        return evaluate_motif(data, file, task)

    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model predictions for various chemistry tasks"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "desc",
            "forward",
            "retro",
            "property",
            "naming",
            "propqc2d",
            "propqc3d",
            "motif",
        ],
        help="Task to evaluate",
    )

    parser.add_argument(
        "--path",
        type=str,
        default="/hub_data5/jinyoungp/all_checkpoints/newLLaMo_epoch3_epoch3_ft_",
        help="Base path to the model checkpoints",
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all tasks")

    args = parser.parse_args()

    # Initialize tokenizer if needed for the task
    tokenizer = None
    if args.task in ["desc", "naming"] or args.all:
        tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_fast=False
        )

    # Run evaluation
    if args.all:
        final_results = {}
        # for task in ["naming"]:

        # for task in ["desc", "prop", "prop1d", "propdesc", "motif"]:
        for task in ["desc", "prop1d", "propdesc", "motif"]:
            for num in range(10):
                file_name = (
                    args.path
                    + "_"
                    + task
                    + f"/lightning_logs/version_{num}/predictions.txt"
                )
                if os.path.exists(file_name):
                    break
            if task == "prop":
                task = "propqc3d"
            elif task == "prop1d":
                task = "propqc2d"

            results = evaluate_task(task=task, file=file_name, tokenizer=tokenizer)
            final_results.update(results)

        print("================================")
        print("Final Results")
        print("================================")
        for metric, value in final_results.items():
            print(f"{metric}: {value:.4f}")

            # print(f"\nResults for {task} task:")
            # if isinstance(results, dict):
            #     for metric, value in results.items():
            #         print(f"{metric}: {value:.4f}")
            # else:
            #     import pdb; pdb.set_trace()
            #     print(results)
    else:
        results = evaluate_task(task=args.task, file=args.path, tokenizer=tokenizer)

        # Print results
        print(f"\nResults for {args.task} task:")
        if isinstance(results, dict):
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")
        else:
            print(results)
