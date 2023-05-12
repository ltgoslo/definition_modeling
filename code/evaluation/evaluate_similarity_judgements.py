import argparse
import csv
import logging
import os.path
from collections import defaultdict
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def main(args):
    gold_df = pd.read_csv(
        args.gold_judgements_path, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
    )
    if args.debug_instances:
        gold_df = gold_df[:args.debug_instances]

    assert not gold_df.isnull().values.any()

    id_pair2gold = defaultdict(lambda: defaultdict(float))
    for _, row in gold_df.iterrows():
        word, id1, id2, score = row["Lemma"], row["Usage0"], row["Usage1"], row["Score"]
        word = word.split("_")[0]
        id_pair2gold[word][(id1, id2)] = float(score)
        id_pair2gold["ALL"][(id1, id2)] = float(score)

    if os.path.isfile(args.predictions_path):
        if not (args.predictions_path.endswith(".sim.tsv") or args.predictions_path.endswith(".sim.tsv.gz")):
            raise ValueError("Invalid path.")
        predictions_paths = [args.predictions_path]
    elif os.path.isdir(args.predictions_path):
        predictions_paths = args.predictions_path

    all_scores = list()
    for filename in os.listdir(predictions_paths):
        if not (filename.endswith(".sim.tsv") or filename.endswith(".sim.tsv.gz")):
            continue
        preds_df = pd.read_csv(
            os.path.join(args.predictions_path, filename), delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
        )
        assert not preds_df.isnull().values.any()
        metrics = preds_df.columns[3:].tolist()

        id_pair2score = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        for _, row in preds_df.iterrows():
            word, id1, id2 = row["word"], row["id1"], row["id2"]
            for metric in metrics:
                id_pair2score[word][metric][(id1, id2)] = float(row[metric])
                id_pair2score["ALL"][metric][(id1, id2)] = float(row[metric])

        for word in id_pair2gold:
            for metric in metrics:
                pred_scores = []
                gold_scores = []
                for (id1, id2) in id_pair2gold[word]:
                    pred_scores.append(id_pair2score[word][metric][(id1, id2)])
                    gold_scores.append(id_pair2gold[word][(id1, id2)])

                r, r_p_val = spearmanr(pred_scores, gold_scores, nan_policy="raise")
                tau, tau_pval = kendalltau(pred_scores, gold_scores, nan_policy="raise")
                # logger.info("{} {}   r = {:.3f} p = {:.3f}   tau = {:.3f} p = {:.3f}".format(
                #     word, metric, r, r_p_val, tau, tau_pval))

                all_scores.append((filename, word, metric, r, r_p_val, tau, tau_pval))

    all_scores_df = pd.DataFrame(
        all_scores,
        columns=["method", "target", "metric", "spearman_r", "spearman_pval", "kendall_tau", "kendall_pval"]
    )

    all_scores_df.to_csv(args.output_path, sep="\t", quoting=csv.QUOTE_NONE, index=False)
    logger.info(f"Correlation scores saved: {args.output_path}")


if __name__ == '__main__':

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--gold_judgements_path",
        "-g",
        help="Path to the human pairwise judgements",
        required=True,
    )
    arg(
        "--predictions_path",
        "-p",
        help="Path to the predicted pairwise judgements",
        required=False
    )
    arg(
        "--output_path",
        "-o",
        help="Where to save the correlation scores",
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