#!/bin/env python3
# coding: utf-8

import argparse
import logging
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--labels", "-c", help="Path to the TSV file with sense clusters and their labels",
        required=True)
    arg("--periods", "-p", help="Path to the TSV file with sense clusters, ids and periods",
        required=True)
    arg("--emb", "-e", help="Path to the NPZ file with sense label embeddings", required=True)
    args = parser.parse_args()

    z_threshold = 1  # minimum z-value for the cluster pair to be linked
    embeddings = np.load(args.emb)
    data = pd.read_csv(args.periods, delimiter="\t", header=0)
    labels = pd.read_csv(args.labels, delimiter="\t", header=0)

    logger.info(embeddings)
    logger.info(data)
    logger.info(labels)

    print("word\tedge\tcluster pair (sense_period)\tsimilarity\tlabel1\tlabel2")
    for w in data.word.unique():
        df = data[data.word == w]
        df_labels = labels[labels.Targets == w]
        valid_clusters = df_labels.Clusters.unique()
        logger.debug(f"Valid clusters: {valid_clusters}")
        logger.info(w)
        senses = {}
        examples = {}
        for row in df.iterrows():
            period = row[1].period
            cluster = row[1].cluster
            usage = row[1].id
            context = row[1].Real_Contexts
            definition = row[1].Definitions
            if cluster not in valid_clusters:
                continue
            cluster_name = str(cluster) + "_" + str(period)
            real_label = df_labels[df_labels.Clusters == cluster].Definitions.values[0]
            logger.debug(f"{cluster_name}\t{real_label}")
            if cluster_name not in senses:
                senses[cluster_name] = real_label
            if cluster_name not in examples:
                examples[cluster_name] = []
            examples[cluster_name].append(context)
        logger.info(senses)
        distances = {}
        for pair in combinations(senses, 2):
            if pair[0].split("_")[0] == pair[1].split("_")[0]:
                continue
            distance = np.dot(embeddings[senses[pair[0]]], embeddings[senses[pair[1]]])
            distances[pair] = distance
        if distances:
            dist_values = [distances[el] for el in distances]
            zscores = stats.zscore(dist_values, axis=None)
            dist_values = dict(zip(dist_values, zscores))
            logger.debug(dist_values)
            for el in distances:
                label0 = senses[el[0]]
                label1 = senses[el[1]]
                if dist_values[distances[el]] > z_threshold:
                    print(f"{w}\tLINK\t{el}\t{distances[el]}\t{label0}\t{label1}")
                    logger.info(examples[el[0]])
                    logger.info(examples[el[1]])
                else:
                    print(f"{w}\tNOLINK\t{el}\t{distances[el]}\t{label0}\t{label1}")
