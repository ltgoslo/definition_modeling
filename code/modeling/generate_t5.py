#!/bin/env python3
# coding: utf-8
import argparse
import csv
import logging
from os import path
import pandas as pd
import torch
import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


def load_data(path_to_data, split="test"):
    if path.isfile(path_to_data):
        logger.info(f"The input data is a file: {path_to_data}")
        df = pd.read_csv(
            path_to_data,
            delimiter="\t",
            header="infer",
            quoting=csv.QUOTE_NONE,
            encoding="utf-8",
            on_bad_lines="warn",
        )
        if len(df.columns) == 2:
            df.columns = ["Targets", "Context"]
        logger.info(f"Found the following data columns: {df.columns}")
    else:
        logger.info(f"The input data is a directory: {path_to_data}")
        if "oxford" not in path_to_data and "wordnet" not in path_to_data:
            if split:
                datafile = path.join(path_to_data, f"{split}.complete.tsv.gz")
            else:
                datafile = path.join(path_to_data, f"complete.tsv.gz")
            df = pd.read_csv(
                datafile,
                delimiter="\t",
                header=0,
                quoting=csv.QUOTE_NONE,
                encoding="utf-8",
                on_bad_lines="warn",
            )
            df["Context"] = df.example
            df["Targets"] = [w.split("%")[0] for w in df.word]
            try:
                df["Definition"] = df.gloss
            except AttributeError:
                logger.info("No definitions found in the input file")
        else:
            datafile = path.join(path_to_data, split + ".eg.gz")
            datafile_defs = path.join(path_to_data, split + ".txt.gz")
            df = pd.read_csv(
                datafile,
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
                encoding="utf-8",
                on_bad_lines="warn",
            )
            df_defs = pd.read_csv(
                datafile_defs,
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
                encoding="utf-8",
                on_bad_lines="warn",
            )
            df_defs.columns = [
                "Sense",
                "Ignore1",
                "Ignore2",
                "Definition",
                "Ignore3",
                "Ignore4",
            ]
            df.columns = ["Sense", "Context"]
            df["Targets"] = [w.split("%")[0] for w in df.Sense]
            df["Definition"] = df_defs.Definition
    if "wordnet" in path_to_data:
        df["POS"] = [w.split("%")[1].split(".")[2] for w in df.Sense]
    contexts = [
        ctxt.replace("<TRG>", targetword).strip()
        for ctxt, targetword in zip(df.Context, df.Targets)
    ]
    df["Real_Contexts"] = contexts
    return df


def define(
        in_prompts,
        lm,
        cur_tokenizer,
        arguments,
        targets,
        filter_target=False,
        num_beams=1,
        num_beam_groups=1,
        sampling=False,
        temperature=1.0,
        repetition_penalty=1.0,
):
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

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        target_ids = target_ids.to("cuda")

    test_dataset = torch.utils.data.TensorDataset(
        inputs["input_ids"], inputs["attention_mask"], target_ids
    )
    test_iter = torch.utils.data.DataLoader(
        test_dataset, batch_size=arguments.bsize, shuffle=False
    )
    logger.info(f"Generating definitions with batch size {arguments.bsize}...")
    gen_args = dict(
        do_sample=sampling,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
    if num_beam_groups > 1:
        gen_args["diversity_penalty"] = 0.5
    definitions = []
    for inp, att, targetwords in tqdm.tqdm(test_iter):
        if filter_target:
            bad = [[el] for el in targetwords.tolist()]
            outputs = lm.generate(
                input_ids=inp,
                attention_mask=att,
                max_new_tokens=60,
                bad_words_ids=bad,
                **gen_args,
            )
        else:
            outputs = lm.generate(
                input_ids=inp, attention_mask=att, max_new_tokens=60, **gen_args
            )
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
    arg(
        "--testdata",
        "-t",
        help="Path to the directory or the file with the input data",
        required=True,
    )
    arg("--bsize", "-b", type=int, help="Batch size", default=4)
    arg("--maxl", "-ml", type=int, help="Max source length", default=256)
    arg(
        "--save",
        "-s",
        help="Where to save the predicted definitions",
        default="predicted.tsv.gz",
    )
    arg(
        "--prompt",
        "-p",
        type=int,
        help="Prompt to use, from the prompt list below",
        default=8,
    )
    arg(
        "--filter",
        "-f",
        type=int,
        help="Filter out target word from definitions?",
        choices=[0, 1],
        default=1,
    )
    arg(
        "--sampling",
        "-smpl",
        type=int,
        help="Sampling instead of greedy decoding",
        choices=[0, 1],
        default=0,
    )
    arg("--rpenalty", "-rep", type=float, help="Repetition penalty", default=1.0)
    arg(
        "--num_beams",
        "-beams",
        type=int,
        help="Number of beams for beam search",
        default=1,
    )
    arg(
        "--num_beam_groups",
        "-bg",
        type=int,
        help="Number of beam groups for beam search",
        default=1,
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    if torch.cuda.is_available():
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, device_map="auto")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model, low_cpu_mem_usage=True
        )

    logger.info(f"Model loaded from {args.model}")

    test_dataframe = load_data(
        args.testdata, split="trial"
    )  # Don't forget to choose the correct split

    prompts = [
        ["", "post"],  # 0
        ["Give the definition of <TRG>:", "pre"],  # 1
        ["Define <TRG>:", "pre"],  # 2
        ["Define the word <TRG>:", "pre"],  # 3
        ["What is the definition of <TRG>?", "pre"],  # 4
        ["Give the definition of <TRG>", "post"],  # 5
        ["Define <TRG>", "post"],  # 6
        ["Define the word <TRG>", "post"],  # 7
        ["What is the definition of <TRG>?", "post"],  # 8
        ["Quelle est la définition de <TRG>?", "post"],  # 9
        ["Что такое <TRG>?", "post"],  # 10
        ["Hva betyr <TRG>?", "post"],  # 11
        ["Was ist die Definition von <TRG>?", "post"],  # 12
    ]

    task_instructions = [prompts[args.prompt]]

    for task_prefix in task_instructions:
        logger.info(f"Generating with the task instruction {task_prefix}...")
        identifier = "_".join(task_prefix).lower().replace(" ", "_")
        input_sentences = []
        for target, context in zip(
                test_dataframe.Targets, test_dataframe.Real_Contexts
        ):
            if task_prefix[1] == "pre":
                prompt = " ".join([task_prefix[0].replace("<TRG>", target), context])
            else:
                prompt = " ".join([context, task_prefix[0].replace("<TRG>", target)])
            input_sentences.append(prompt)
        answers = define(
            input_sentences,
            model,
            tokenizer,
            args,
            test_dataframe.Targets.tolist(),
            filter_target=args.filter,
            sampling=args.sampling,
            repetition_penalty=args.rpenalty,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
        )

        test_dataframe["Generated_Definition"] = answers
        if "CoDWoE" in args.testdata:
            test_dataframe = test_dataframe[
                ["Targets", "Real_Contexts", "Definition", "Generated_Definition"]
            ]
        test_dataframe["Real_Contexts"] = input_sentences
        test_dataframe.to_csv(
            args.save, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_NONE
        )
        logger.info(f"Predictions of {identifier} saved to {args.save} ...")
