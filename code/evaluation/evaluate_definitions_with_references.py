import argparse
import csv
import re
import evaluate
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from nltk.translate import bleu_score, nist_score
from tqdm import tqdm
# from moverscore import word_mover_score


# idf_dict_hyp = defaultdict(lambda: 1.)
# idf_dict_ref = defaultdict(lambda: 1.)


def get_rid_of_period(l):
    pattern = re.compile("\.(?!\d)")
    return [pattern.sub('', sent) for sent in l]


def main(args):
    if os.path.isdir(args.gold_path):
        gold_dictionary_path = os.path.join(args.gold_path, "test.txt.gz")
    else:
        gold_dictionary_path = args.gold_path

    if "CoDWoE" in gold_dictionary_path:
        gold_dataframe = pd.read_csv(
            gold_dictionary_path,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            encoding='utf-8',
        )
        gold_dataframe = gold_dataframe.rename(columns={"word": "Sense", "gloss": "Definition"})
    else:
        gold_dataframe = pd.read_csv(
            gold_dictionary_path,
            delimiter="\t",
            header=0,
            usecols=[0, 3],
            names=["Sense", "Definition"],
            quoting=csv.QUOTE_NONE,
            encoding='utf-8'
        )
    gold_dictionary = defaultdict(list)
    for sense_id, gold_definition in zip(gold_dataframe.Sense, gold_dataframe.Definition):
        if type(gold_definition) is not str:
            raise NotImplementedError(
                f"Gold definition field should contain a string: {type(gold_definition)}, {gold_definition}")
        gold_dictionary[sense_id].append(gold_definition)

    eval = {
        # metric: (evaluator, output_key)
        "rougeL": (evaluate.load("rouge"), "rougeL"),
        "meteor": (evaluate.load("meteor"), "meteor"),
        "bertscore": (evaluate.load("bertscore"), "f1"),
        "sacrebleu": (evaluate.load("sacrebleu"), "score"),
        "exact_match": (evaluate.load("exact_match"), "exact_match"),
        # "bleu": (evaluate.load("bleu"), "bleu"),
    }

    if not args.metrics:
        args.metrics = ["rougeL", "nltk_bleu", "nist", "sacrebleu", "meteor", "bertscore", "exact_match", "mauve"]

    if "mauve" in args.metrics:
        mauve = evaluate.load("mauve")

    if os.path.isfile(args.preds_path) and (args.preds_path.endswith('.tsv') or args.preds_path.endswith('.csv')):
        prediction_files = [args.preds_path]
    elif os.path.isdir(args.preds_path):
        prediction_files = [os.path.join(args.preds_path, f) for f in os.listdir(args.preds_path) if f.endswith(".tsv")]
    else:
        raise ValueError()

    if args.dataset_name:
        prediction_files = [f for f in prediction_files if args.dataset_name in f]

    for prediction_file in tqdm(prediction_files):
        output_file_path = os.path.join(
            args.output_dir,
            prediction_file.split("/")[-1].split(".tsv")[0] + ".eval.tsv"
        )
        if os.path.exists(output_file_path) and not args.overwrite:
            print(f"Output file exists: {output_file_path}")
            continue
        print(prediction_file)

        preds_dataframe = pd.read_csv(prediction_file, delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
        scores = defaultdict(list)
        count = 0
        prev_sense_id, same_sense_id_count = "", 0

        if "mauve" in args.metrics:
            preds_for_mauve, gold_for_mauve = [], []

        if "CoDWoE" in prediction_file:
            sense_ids, predicted_definitions = preds_dataframe.Targets, preds_dataframe.Definitions
        else:
            sense_ids, predicted_definitions = preds_dataframe.Sense, preds_dataframe.Definitions

        for sense_id, predicted_definition in zip(sense_ids, predicted_definitions):
            if type(predicted_definition) is list:
                raise NotImplementedError("Evaluation of multiple samples not implemented yet.")
            if type(predicted_definition) is not str:
                predicted_definition = ""

            if sense_id == prev_sense_id:
                same_sense_id_count += 1
            else:
                prev_sense_id = sense_id
                same_sense_id_count = 0

            pred_def = get_rid_of_period([predicted_definition])[0]
            gold_definitions = get_rid_of_period(gold_dictionary[sense_id])

            if "mauve" in args.metrics:
                if pred_def:
                    preds_for_mauve.append(pred_def)
                else:
                    preds_for_mauve.append("</s>")
                gold_for_mauve.append(gold_definitions[same_sense_id_count])

            count += 1
            if args.debug_instances and (count > args.debug_instances):
                break

            pred_def_scores = defaultdict(list)
                # if len(gold_definitions) > 1:
                #     print(sense_id, pred_def, gold_definitions)

            for gold_def in gold_definitions:
                for metric in args.metrics:
                    if metric == "mauve":
                        continue
                    if metric == "nltk_bleu":
                        auto_reweigh = False if len(pred_def.split()) == 0 else True
                        pred_def_scores["nltk_bleu"].append(bleu_score.sentence_bleu(
                            gold_def.split(),
                            pred_def.split(),
                            smoothing_function=bleu_score.SmoothingFunction().method2,
                            auto_reweigh=auto_reweigh
                        ))
                    elif metric == "nist":
                        n = 5
                        pred_len = len(pred_def.split())
                        if pred_len < 5:
                            n = pred_len
                        try:
                            pred_def_scores["nist"].append(nist_score.sentence_nist(
                                gold_def.split(),
                                pred_def.split(),
                                n=n
                            ))
                        except:
                            pred_def_scores["nist"].append(0)
                    elif metric == "bertscore":
                        evaluator, output_key = eval[metric]
                        pred_def_scores[metric].append(evaluator.compute(
                            predictions=[pred_def], references=[gold_def], lang="en")[output_key]
                        )
                    else:
                        evaluator, output_key = eval[metric]
                        pred_def_scores[metric].append(evaluator.compute(
                            predictions=[pred_def], references=[gold_def])[output_key]
                        )

            pred_def_score = dict()
            for metric in args.metrics:
                if not pred_def_scores[metric]:
                    pred_def_score[metric] = 0.
                elif args.multiple_definitions_same_sense_id == "max":
                    pred_def_score[metric] = np.max(pred_def_scores[metric])
                elif args.multiple_definitions_same_sense_id == "mean":
                    pred_def_score[metric] = np.mean(pred_def_scores[metric])
                elif args.multiple_definitions_same_sense_id == "ignore":
                    pred_def_score[metric] = pred_def_scores[metric][same_sense_id_count]
                scores[metric].append(pred_def_score[metric])

        if "mauve" in args.metrics:
            scores["mauve"].append(mauve.compute(
                predictions=preds_for_mauve,
                references=gold_for_mauve,
                featurize_model_name="gpt2",
                max_text_length=512).mauve
            )

        aggr_output = []
        for metric in args.metrics:
            aggr_output.append((metric, np.mean(list(scores[metric]))))
            if args.save_instance_scores and metric != "mauve":
                output_file_path_metric = output_file_path.split(".tsv")[0] + f".{metric}.tsv"
                if os.path.exists(output_file_path_metric) and not args.overwrite:
                    print(f"Metric output file exists: {output_file_path_metric}")
                    continue
                with open(output_file_path_metric, "w") as f_out:
                    for score in scores[metric]:
                        print(score, file=f_out)

        with open(output_file_path, "w") as f_out:
            for (metric, score) in aggr_output:
                print("\t".join([metric, "{:.4f}".format(score)]), file=f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds_path",
        type=str,
        help="Predictions' file or directory path. If directory, loops through tsv files.",
    )
    parser.add_argument(
        "--gold_path",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
    )
    parser.add_argument(
        "--metrics",
        nargs='*',
    )
    parser.add_argument(
        "--multiple_definitions_same_sense_id",
        type=str,
        default='max',
        help="Options are: 'max', 'mean', 'ignore'.",
    )
    parser.add_argument(
        "--debug_instances",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true"
    )
    parser.add_argument(
        "--save_instance_scores",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)
