#!/bin/env python3
# coding: utf-8
import csv
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import logging
import tqdm
from os import path


def load_data(path_to_data, split="test"):
    if "CoDWoE" in path_to_data or "dwug" in path_to_data:
        datafile = path.join(path_to_data, "complete.tsv.gz")
        df = pd.read_csv(datafile, delimiter="\t", header=0, quoting=csv.QUOTE_NONE,
                         encoding="utf-8", on_bad_lines="warn")
        df["Context"] = df.example
        df["Targets"] = [w.split("%")[0] for w in df.word]
        try:
            df["Definition"] = df.gloss
        except AttributeError:
            print("No definitions found, replacing them with examples")
            df["Definition"] = df.example
    else:
        datafile = path.join(path_to_data, split + ".eg.gz")
        datafile_defs = path.join(path_to_data, split + ".txt.gz")
        df = pd.read_csv(datafile, delimiter="\t", quoting=csv.QUOTE_NONE,
                         encoding="utf-8", on_bad_lines="warn")
        df_defs = pd.read_csv(datafile_defs, delimiter="\t", quoting=csv.QUOTE_NONE,
                         encoding="utf-8", on_bad_lines="warn")
        df_defs.columns = ["Sense", "Ignore1", "Ignore2", "Definition", "Ignore3", "Ignore4"]
        df.columns = ["Sense", "Context"]
        df["Targets"] = [w.split("%")[0] for w in df.Sense]
        df["Definition"] = df_defs.Definition
    if "wordnet" in path_to_data:
        df["POS"] = [w.split("%")[1].split(".")[2] for w in df.Sense]
    contexts = [ctxt.replace("<TRG>", targetword).strip() for ctxt, targetword
                in zip(df.Context, df.Targets)]
    df["Real_Contexts"] = contexts
    return df


def define(in_prompts, lm, cur_tokenizer, arguments, targets, filter_target=False):
    logger.info(f"Tokenizing with max length {arguments.maxl}...")
    inputs = cur_tokenizer(
        in_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=arguments.maxl,
    )
    logger.info("Tokenizing finished.")

    target_ids = cur_tokenizer(targets, add_special_tokens=False).input_ids
    target_ids = torch.tensor([el[-1] for el in target_ids])

    test_dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"],
                                                  target_ids)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=arguments.bsize, shuffle=False)
    logger.info(f"Generating definitions with batch size {arguments.bsize}...")

    definitions = []
    for inp, att, targetwords in tqdm.tqdm(test_iter):
        if filter_target:
            bad = [[el] for el in targetwords.tolist()]
            outputs = lm.generate(input_ids=inp, attention_mask=att,
                                  do_sample=False, bad_words_ids=bad)
        else:
            outputs = lm.generate(input_ids=inp, attention_mask=att,
                                  do_sample=False)  # Generate multiple definitions?
        predictions = cur_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        definitions += predictions
    logger.info(f"Generating definitions finished")
    return definitions


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--model", "-m", help="Path to or name of a T5 model", required=True)
    arg("--testdata", "-t", help="Path to the directory with the test data (2 files)",
        required=True)
    arg("--bsize", "-b", type=int, help="Batch size", default=4)
    arg("--maxl", "-ml", type=int, help="Max source length", default=256)
    arg("--save", "-s", help="Where to save the predicted definitions",
        default="predicted.tsv.gz")
    arg("--prompt", "-p", type=int, help="Prompt to use, from the prompt list below "
                                         "(from 0 to 7)", default=7)
    arg("--filter", "-f", type=int, help="Filter out target word from definitions?", choices=[0, 1],
        default=0)

    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model, add_prefix_space=True)
    if torch.cuda.is_available():
        model = T5ForConditionalGeneration.from_pretrained(args.model, device_map="auto")
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model, low_cpu_mem_usage=True)

    logger.info(f"Model loaded from {args.model}")

    test_dataframe = load_data(args.testdata, split="test")

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

    task_instructions = [prompts[args.prompt]]

    for task_prefix in task_instructions:
        logger.info(f"Generating with the task instruction {task_prefix}...")
        identifier = "_".join(task_prefix).lower().replace(" ", "_")
        input_sentences = []
        for target, context in zip(test_dataframe.Targets, test_dataframe.Real_Contexts):
            if task_prefix[1] == "pre":
                prompt = " ".join([task_prefix[0].replace("<TRG>", target), context])
            else:
                prompt = " ".join([context, task_prefix[0].replace("<TRG>", target)])
            input_sentences.append(prompt)
        answers = define(input_sentences, model, tokenizer, args, test_dataframe.Targets.tolist(),
                         filter_target=args.filter)

        test_dataframe["Definitions"] = answers
        outname = "_".join([identifier, args.save])
        if "CoDWoE" in args.testdata:
            test_dataframe = test_dataframe[["Targets", "Real_Contexts", "Definitions"]]
        test_dataframe.to_csv(outname, sep="\t", index=False, encoding="utf-8")
        logger.info(f"Predictions saved to {outname} ...")
