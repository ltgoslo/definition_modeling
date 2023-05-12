#!/bin/env python3
# coding: utf-8

import argparse
import logging
from os import path, walk
import pandas as pd
import numpy as np

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

    usage0 = []
    usage1 = []
    score = []
    words = []
    for root, dirs, files in walk(args.data):
        # if el.name.endswith(".csv"):
        for f in files:
            if f == "judgments.csv":
                logging.info(f"Processing {root}...")
                dataframe = pd.read_csv(path.join(root, f), delimiter="\t")
                judgments = {}
                for index, row in dataframe.iterrows():
                    pair = (row.identifier1, row.identifier2)
                    lemma = row.lemma
                    if (row.identifier2, row.identifier1) in judgments:
                        pair = (row.identifier2, row.identifier1)
                    if pair not in judgments:
                        judgments[pair] = []
                    judgments[pair].append(row.judgment)

                for el in judgments:
                    if judgments[el].count(0) >= (len(judgments[el]) / 2):
                        logger.info(f"Removing {el} because of 0 judgments")
                        continue
                    average = np.mean(judgments[el])
                    usage0.append(el[0])
                    usage1.append(el[1])
                    score.append(average)
                    words.append(lemma)

    df = pd.DataFrame(list(zip(words, usage0, usage1, score)),
                      columns=["Lemma", "Usage0", "Usage1", "Score"])
    df.to_csv("dwug_en_pairwise_judgments_test.tsv", sep="\t", index=False)
