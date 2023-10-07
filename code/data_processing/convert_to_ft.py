#!/bin/env python3
# coding: utf-8

import argparse
import logging
from os import path
import pandas as pd
from generate_flan import load_data

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--traindata", "-t", help="Path to the directory with the train data",
        required=True)
    arg("--save", "-s", help="Where to save the resulting file",
        default="ft_data.tsv.gz")
    arg("--prompt", "-p", type=int, help="Prompt to use, from the prompts list below", default=7)

    args = parser.parse_args()

    prompts = [
        ["Give the definition of <TRG>:", "pre"],
        ["Define <TRG>:", "pre"],
        ["Define the word <TRG>:", "pre"],
        ["What is the definition of <TRG>?", "pre"],
        ["Give the definition of <TRG>", "post"],
        ["Define <TRG>", "post"],
        ["Define the word <TRG>", "post"],
        ["What is the definition of <TRG>?", "post"],
        ["Quelle est la définition de <TRG>?", "post"],
        ["Что такое <TRG>?", "post"],
    ]

    dataframe = load_data(args.traindata)

    task_instruction = prompts[args.prompt]

    input_sentences = []
    for target, context in zip(dataframe.Targets, dataframe.Real_Contexts):
        if task_instruction[1] == "pre":
            prompt = ". ".join([task_instruction[0].replace("<TRG>", target), context])
        else:
            prompt = ". ".join([context, task_instruction[0].replace("<TRG>", target)])
        input_sentences.append(prompt)

    dataframe["examples"] = input_sentences

    logging.info("Finished reading and processing examples.")

    if "oxford" in args.traindata or "wordnet" in args.traindata:
        definitionfile = path.join(args.traindata, "valid.txt.gz")
        df = pd.read_csv(definitionfile, delimiter="\t")
        df.columns = ["Sense", "POS", "Dataset", "Definitions", "Dummy0", "Dummy1"]
        dataframe["definitions"] = df.Definitions
    else:
        dataframe["definitions"] = dataframe.gloss
    logging.info("Finished reading and processing definitions.")

    dataframe = dataframe[["examples", "definitions"]]
    total_count = len(dataframe.index)
    len_train = int(total_count * 0.9)
    train_dataframe = dataframe.iloc[:len_train]
    val_dataframe = dataframe.iloc[len_train:]

    train_dataframe.to_csv(f"{args.save}_train.tsv.gz", sep="\t", index=False)
    val_dataframe.to_csv(f"{args.save}_val.tsv.gz", sep="\t", index=False)
    logger.info(f"Training data saved to {args.save} ...")
