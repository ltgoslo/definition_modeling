import argparse
import os
import pandas as pd
import csv


def main(args):
    target_aggr_info = []  #[("id", "word", "pos", "date", "period", "cluster", "example")]
    clusters_dir = os.path.join(args.dwug_dir, "clusters/opt")
    for filename in os.listdir(clusters_dir):
        f = os.path.join(clusters_dir, filename)
        if not f.endswith("csv"):
            continue

        target = filename.split(".")[0]
        target_word = target.split("_")[0]

        print(target_word)

        id2cluster = dict()
        for _, row in pd.read_csv(f, delimiter="\t", quoting=csv.QUOTE_NONE).iterrows():
            id2cluster[row["identifier"]] = row["cluster"]

        uses_filepath = os.path.join(args.dwug_dir, f"data/{target}/uses.csv")
        for _, row in pd.read_csv(uses_filepath, delimiter="\t", quoting=csv.QUOTE_NONE).iterrows():
            target_aggr_info.append((
                row["identifier"],
                target_word,
                row["pos"],
                str(row["date"]),
                str(row["grouping"]),
                str(id2cluster[row["identifier"]]) if row["identifier"] in id2cluster else "-1",
                row["indexes_target_token"].strip(),
                row["context"].strip()
            ))

    output_df = pd.DataFrame(
        target_aggr_info, columns=["id", "word", "pos", "date", "period", "cluster", "target_indices", "example"]
    )
    output_df.to_csv(args.output_path, sep="\t", quoting=csv.QUOTE_NONE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dwug_dir",
        type=str,
        help="DWUG directory (e.g., `dwug_en/`). It should contain the folders `clusters` and `data`..",
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    args = parser.parse_args()
    main(args)