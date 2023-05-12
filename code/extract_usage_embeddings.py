import argparse
import csv
import itertools
import logging
from collections import defaultdict

import evaluate
import pandas as pd
import torch
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def get_target_embedding(hidden_states, start_index, end_index, pooling="mean", representation="token"):
    embedded_sequence = hidden_states.squeeze()
    if representation == "token":
        target_embeddings = embedded_sequence[start_index: end_index]
    elif representation == "sentence":
        target_embeddings = embedded_sequence
    else:
        raise ValueError("Representation type not available. Available types: 'token', 'sentence'.")
    if pooling == "mean":
        target_embedding = target_embeddings.mean(dim=0)
    elif pooling == "max":
        target_embedding, _ = target_embeddings.max(dim=0)
    elif pooling == "first":
        target_embedding = target_embeddings[0]
    elif pooling == "last":
        target_embedding = target_embeddings[-1]
    else:
        raise ValueError("Pooling strategy not available. Available strategies: 'mean', 'max', 'first', 'last'.")
    if np.isnan(target_embedding).any():
        raise ValueError()
    return target_embedding


def find_tokens(indices_in_sentence, offset_mapping):
    original_start_idx, original_end_idx = indices_in_sentence
    i, j = 0, len(offset_mapping) - 1
    tokens_start_idx, tokens_end_idx = None, None
    while i <= j and (tokens_start_idx is None or tokens_end_idx is None):
        if offset_mapping[i, 0] <= original_start_idx < offset_mapping[i, 1] and offset_mapping[i].sum() > 0:
            tokens_start_idx = i
        else:
            i += 1
        if offset_mapping[j, 0] < original_end_idx <= offset_mapping[j, 1] and offset_mapping[j].sum() > 0:
            tokens_end_idx = j
        else:
            j -= 1
    tokens_end_idx = tokens_end_idx + 1 if tokens_end_idx else None
    return tokens_start_idx, tokens_end_idx


def main(args):
    data_df = pd.read_csv(
        args.data_path, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
    )
    if args.debug_instances:
        data_df = data_df[:args.debug_instances]

    word2ids = defaultdict(list)
    word2usages = defaultdict(list)
    word2str_indices = defaultdict(list)
    for _, row in data_df.iterrows():
        word2ids[row["word"]].append(row["id"])
        word2usages[row["word"]].append(row["example"])
        word2str_indices[row["word"]].append(list(map(int, row["target_indices"].split(":"))))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)

    embeddings = {}
    for word in tqdm(sorted(word2ids.keys())):
        inputs = tokenizer(
            word2usages[word],
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True
        )
        with torch.no_grad():
            model_output = model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                output_hidden_states=True
            )

        for i in range(len(word2ids[word])):
            if args.representation == "token":
                start_idx, end_idx = find_tokens(word2str_indices[word][i], inputs.offset_mapping[i])
                print(tokenizer.convert_ids_to_tokens(inputs.input_ids[i][start_idx:end_idx]))
            else:
                start_idx, end_idx = None, None
            embeddings[word2ids[word][i]] = get_target_embedding(
                model_output.hidden_states[-1][i], start_idx, end_idx, args.pooling, args.representation
            )

    np.savez_compressed(args.output_path, **embeddings)


if __name__ == '__main__':

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--model",
        "-m",
        help="Path to a Huggingface model",
        default="roberta-large",
    )
    arg(
        "--data_path",
        "-d",
        help="Path to the `complete.tsv.gz` file containing usages and target word indices",
        required=True,
    )
    arg(
        "--output_path",
        "-o",
        help="Where to save the usage embeddings",
    )
    arg(
        "--debug_instances",
        "-db",
        type=int,
        default=None,
        help="The number of embeddings to embed. If None, all definitions are embedded.",
    )
    arg(
        "--pooling",
        "-p",
        type=str,
        default="mean",
        help="The pooling strategy for multi-token targets: 'mean', 'max', 'first', 'last'.",
    )
    arg(
        "--representation",
        "-r",
        type=str,
        default="token",
        help="The type of representation to extract: 'token' or 'sentence",
    )
    args = parser.parse_args()
    main(args)
