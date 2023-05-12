#!/bin/env python3
# coding: utf-8

import argparse
import logging
from os import path
import pandas as pd
from generate_t5 import load_data

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
    arg("--prompt", "-p", type=int, help="Prompt to use, from the prompt list below "
                                         "(from 0 to 7)", default=7)

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
    ]

    train_dataframe = load_data(args.traindata)

    task_instruction = prompts[args.prompt]

    input_sentences = []
    for target, context in zip(train_dataframe.Targets, train_dataframe.Real_Contexts):
        if task_instruction[1] == "pre":
            prompt = ". ".join([task_instruction[0].replace("<TRG>", target), context])
        else:
            prompt = ". ".join([context, task_instruction[0].replace("<TRG>", target)])
        input_sentences.append(prompt)

    train_dataframe["examples"] = input_sentences

    logging.info("Finished reading and processing examples.")

    if "oxford" in args.traindata or "wordnet" in args.traindata:
        definitionfile = path.join(args.traindata, "valid.txt.gz")
        df = pd.read_csv(definitionfile, delimiter="\t")
        df.columns = ["Sense", "POS", "Dataset", "Definitions", "Dummy0", "Dummy1"]
        train_dataframe["definitions"] = df.Definitions
    else:
        train_dataframe["definitions"] = train_dataframe.gloss
    logging.info("Finished reading and processing definitions.")

    train_dataframe = train_dataframe[["examples", "definitions"]]
    train_dataframe.to_csv(args.save, sep="\t", index=False)
    logger.info(f"Training data saved to {args.save} ...")
