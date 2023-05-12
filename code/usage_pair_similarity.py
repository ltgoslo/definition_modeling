import argparse
import csv
import itertools
import logging
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, pairwise
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def main(args):
    data_df = pd.read_csv(
        args.data_path, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
    )
    if args.debug_instances:
        data_df = data_df[:args.debug_instances]

    all_embeds = np.load(args.embeddings_path)
    all_ids = list(all_embeds.keys())
    all_embeds_normalised = StandardScaler().fit_transform([all_embeds[i] for i in all_ids])
    all_embeds = {}
    for i, _id in enumerate(all_ids):
        all_embeds[_id] = all_embeds_normalised[i]

    word2ids = defaultdict(list)
    for _, row in data_df.iterrows():
        word2ids[row["word"]].append(row["id"])

    similarity_judgements = []
    for word in tqdm(sorted(word2ids.keys())):
        word_embeds = np.array([all_embeds[i] for i in word2ids[word]])
        # print(word_embeds.shape)
        # word_embeds = StandardScaler().fit_transform(word_embeds)
        # try:
        pairwise_similarities = pairwise.cosine_similarity(word_embeds)
        # except ValueError:
        #     print(word, word_embeds.shape)
        #     print(np.argwhere(np.isnan(word_embeds)))
        idx_pairs = set(itertools.combinations(list(np.arange(len(word2ids[word]))), 2))
        for i, j in idx_pairs:
            similarity_judgements.append((
                word,
                word2ids[word][i],
                word2ids[word][j],
                pairwise_similarities[i][j],
            ))

    df = pd.DataFrame(
        similarity_judgements,
        columns=["word", "id1", "id2", "cosine"]
    )

    if not args.output_path:
        args.output_path = args.embeddings_path.split(".")[0] + ".sim.tsv"
    df.to_csv(args.output_path, sep="\t", quoting=csv.QUOTE_NONE, index=False)


if __name__ == '__main__':

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--embeddings_path",
        "-e",
        help="Path to the .npz file containing usage embeddings",
        required=True,
    )
    arg(
        "--data_path",
        "-d",
        help="Path to the `complete.tsv.gz` file containing usages and ids",
        required=True,
    )
    arg(
        "--output_path",
        "-o",
        help="Where to save the embeddings",
        required=False
    )
    arg(
        "--debug_instances",
        "-db",
        type=int,
        default=None,
        help="The number of definitions to embed. If none, all definitions are embedded.",
    )
    args = parser.parse_args()
    main(args)
