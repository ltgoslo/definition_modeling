#!/bin/env python3
# coding: utf-8

import argparse
import logging
import os.path
from collections import defaultdict
from os import path, walk
import pandas as pd
import numpy as np

def get_annotations(data_path, annotator):
    words, usage0, usage1, score = [], [], [], []
    for root, dirs, files in walk(data_path):
        # if el.name.endswith(".csv"):
        for f in files:
            if f == "judgments.csv":
                # logging.info(f"Processing {root}...")
                dataframe = pd.read_csv(path.join(root, f), delimiter="\t")
                judgments = defaultdict(list)
                for index, row in dataframe.iterrows():
                    if row.annotator != annotator:
                        continue
                    pair = (row.identifier1, row.identifier2)
                    lemma = row.lemma
                    if (row.identifier2, row.identifier1) in judgments:
                        pair = (row.identifier2, row.identifier1)
                    judgments[pair].append(row.judgment)

                for el in judgments:
                    # when multiple judgements are available for a single annotator, use the last round's judgement
                    judgment = judgments[el][-1]
                    if judgment == 0:
                        logger.info(f"Removing {el} because of 0 judgments")
                        continue
                    usage0.append(el[0])
                    usage1.append(el[1])
                    score.append(judgment)
                    words.append(lemma)
    return words, usage0, usage1, score


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--data", "-d", help="Path to the directory with the DWUG data",
        required=True)
    args = parser.parse_args()

    annotators = pd.read_csv(os.path.join(args.data, "annotators.csv"), delimiter="\t").annotator.tolist()
    logger.info(f"{len(annotators)} annotators.")

    for annotator in annotators:
        words, usage0, usage1, score = get_annotations(args.data, annotator)

        df = pd.DataFrame(list(zip(words, usage0, usage1, score)),
                          columns=["Lemma", "Usage0", "Usage1", "Score"])
        df.to_csv(f"pairwise_judgements/pairwise_judgments_{annotator}.tsv", sep="\t", index=False)
