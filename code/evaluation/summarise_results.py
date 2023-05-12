import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
import csv
import os


def main(args):
    scores = defaultdict(dict)

    for filename in os.listdir(os.path.join(args.results_dir)) :#, "other")):
        print(filename)
        if not filename.endswith("eval.tsv"):
            continue

        with open(os.path.join(args.results_dir, f"{filename}"), "r") as f:
            for line in f.readlines():
                metric, score = line.split("\t")
                scores[filename.split(".")[0].replace("_", " ")][metric] = float(score)

        # with open(os.path.join(args.results_dir, f"bertscore/{filename}"), "r") as f:
        #     metric, score = f.readline().split("\t")
        #     scores[filename.split(".")[0].replace("_", " ")][metric] = float(score)

    df_data = []
    for config in scores:
        df_row = [config]
        for metric in scores[config]:
            df_row.append(scores[config][metric])
        df_data.append(df_row)


    df = pd.DataFrame(df_data, columns=["Configuration", "BERTScore", "SacreBLEU", "rougeL", "nltk_BLEU", "NIST", "METEOR"])
    df.sort_values(by="BERTScore", inplace=True, ascending=False)
    df.to_csv(args.output_path, sep="\t", index=False)
    # df = pd.read_csv(os.path.join(args.results_dir, f"other/{f}"), delimiter="\t", header=None).T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    args = parser.parse_args()

    args.results_dir = "eval_output/for_comparison"
    args.output_path = (
        "eval_output/for_comparison/comparison_table.tsv"
    )
    main(args)
