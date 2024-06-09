import argparse
import csv
import os
import re
from collections import defaultdict
import evaluate
import numpy as np
import pandas as pd
from nltk.translate import bleu_score, nist_score


def get_rid_of_period(el):
    pattern = re.compile("\.(?!\d)")
    return [pattern.sub('', sent) for sent in el]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="File with examples, generated and gold definitions",
        required=True
    )
    parser.add_argument(
        "--output",
        type=str,
    )
    parser.add_argument(
        "--metrics",
        nargs='*',
        default=["sacrebleu", "rougeL", "bertscore", "exact_match"]
    )
    parser.add_argument(
        "--multiple_definitions_same_sense_id",
        type=str,
        default="mean",
        help="Options are: 'max', 'mean', 'ignore'.",
    )
    parser.add_argument(
        "--debug_instances",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--save_instance_scores",
        action="store_true"
    )
    parser.add_argument("--whitespace", "-w", type=int,
            help="Use whitespace tokenization instead of default in Rouge?",
            choices=[0, 1], default=1)
    args = parser.parse_args()

    df = pd.read_csv(
        args.data_path,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        encoding='utf-8',
    )

    gold_dictionary = defaultdict(list)
    try:
        sense_ids = df.Sense
    except AttributeError:
        print("No sense ids found, using target words as senses")
        sense_ids = df.Targets

    for sense_id, gold_definition in zip(sense_ids, df.Definition):
        if type(gold_definition) is not str:
            raise NotImplementedError(
                f"Gold definition field should contain a string: {type(gold_definition)}, "
                f"{gold_definition}")
        gold_dictionary[sense_id].append(gold_definition)

    eval_metrics = {
        # metric: (evaluator, output_key)
        "rougeL": (evaluate.load("rouge"), "rougeL"),
        "meteor": (evaluate.load("meteor"), "meteor"),
        "bertscore": (evaluate.load("bertscore"), "f1"),
        "sacrebleu": (evaluate.load("sacrebleu"), "score"),
        "exact_match": (evaluate.load("exact_match"), "exact_match"),
        # "bleu": (evaluate.load("bleu"), "bleu"),
    }

    if not args.metrics:
        args.metrics = ["rougeL", "nltk_bleu", "nist", "sacrebleu", "meteor", "bertscore",
                        "exact_match"]

    if "mauve" in args.metrics:
        mauve = evaluate.load("mauve")

    scores = defaultdict(list)
    count = 0
    prev_sense_id, same_sense_id_count = "", 0

    preds_for_mauve, gold_for_mauve = [], []

    predicted_definitions = df.Generated_Definition

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

        for gold_def in gold_definitions:
            for metric in args.metrics:
                if metric == "mauve":
                    continue
                if metric == "nltk_bleu":
                    auto_reweigh = False if len(pred_def.split()) == 0 else True
                    pred_def_scores[metric].append(bleu_score.sentence_bleu(
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
                    pred_def_scores[metric].append(nist_score.sentence_nist(
                        gold_def.split(),
                        pred_def.split(),
                        n=n
                    ))
                # TODO: Language!!!
                elif metric == "bertscore":
                    evaluator, output_key = eval_metrics[metric]
                    pred_def_scores[metric].append(evaluator.compute(
                        predictions=[pred_def], references=[gold_def], lang="en")[output_key]
                                                  )
                elif metric == "rougeL":
                    evaluator, output_key = eval_metrics[metric]
                    if args.whitespace:
                        pred_def_scores[metric].append(evaluator.compute(predictions=[pred_def],
                            references=[gold_def], tokenizer=lambda x: x.split())[output_key])
                    else:
                        pred_def_scores[metric].append(evaluator.compute(predictions=[pred_def],
                            references=[gold_def],)[output_key])
                else:
                    evaluator, output_key = eval_metrics[metric]
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
            output_file_path_metric = args.output.split(".tsv")[0] + f".{metric}.tsv"
            if os.path.exists(output_file_path_metric) and not args.overwrite:
                print(f"Metric output file exists: {output_file_path_metric}")
                continue
            with open(output_file_path_metric, "w") as f_out:
                for score in scores[metric]:
                    print(score, file=f_out)

    with open(args.output, "w") as f_out:
        for (metric, score) in aggr_output:
            print("\t".join([metric, "{:.4f}".format(score)]), file=f_out)
