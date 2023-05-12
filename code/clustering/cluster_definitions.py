import argparse
import csv
import logging
from collections import defaultdict

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def main(args):
    definitions_df = pd.read_csv(
        args.data_path, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
    )
    all_embeds = np.load(args.embeddings_path)

    word2embeds = defaultdict(list)
    word2labels = defaultdict(list)
    word2ids = defaultdict(list)
    for _, row in definitions_df.iterrows():
        if int(row["cluster"]) == "-1":
            continue
        word2embeds[row["word"]].append(all_embeds[row["id"]])
        word2labels[row["word"]].append(int(row["cluster"]))
        word2ids[row["word"]].append(row["id"])

    ari_scores = {}
    predicted_cluster_assignments = {}
    for word in sorted(word2embeds.keys()):
        X, y = np.array(word2embeds[word]), np.array(word2labels[word])
        K = max(y) + 1
        X = StandardScaler().fit_transform(X)
        if args.pca_dims:
            pca = PCA(n_components=args.pca_dims)
            X = pca.fit_transform(X)
        clustering = KMeans(n_clusters=K, random_state=args.seed).fit(X)
        y_pred = clustering.labels_
        ari_scores[word] = adjusted_rand_score(labels_true=y, labels_pred=y_pred)
        predicted_cluster_assignments[word] = y_pred

    with open(args.output_path, "w") as f:
        print("\t".join(["word", "id", "gold_label", "predicted_label"]), file=f)
        for word in word2ids:
            for i, _id in enumerate(word2ids[word]):
                print("\t".join(list(
                    map(str, [word, _id, word2labels[word][i], predicted_cluster_assignments[word][i]]))),
                    file=f)

    with open(args.output_path + ".ari", "w") as f:
        print("Overall ARI score: {:.3f} ± {:.3f}\n".format(
            np.mean(list(ari_scores.values())), np.std(list(ari_scores.values()))),
            file=f)
        for word, score in ari_scores.items():
            print("{}\t{:.3f}".format(word, score), file=f)


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
        help="Path to a .npz file containing definition embeddings",
        required=True,
    )
    arg(
        "--data_path",
        "-g",
        help="Path to the `complete.tsv.gz` file containing usages and cluster ids",
        required=True,
    )
    arg(
        "--output_path",
        "-o",
        help="Where to save the definition cluster ids",
        required=False
    )
    arg(
        "--debug_instances",
        "-d",
        type=int,
        default=None,
        help="The number of definitions to embed. If none, all definitions are embedded.",
    )
    arg(
        "--pca_dims",
        type=int,
        default=None,
        help="The number of principal components for PCA.",
    )
    arg(
        "--seed",
        type=int,
        default=42,
        help="The random state for K-Means.",
    )
    args = parser.parse_args()
    main(args)
