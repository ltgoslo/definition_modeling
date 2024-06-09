#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import numpy as np
# import seaborn as sns
from transformers import T5Tokenizer
from generate_t5 import load_data

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--model", "-m", help="Path to or name of a T5 model", required=False)
    arg("--data", "-d", help="Path to the directory with the data (2 files)",
        required=True)
    arg("--plot", "-p", help="Whether to show plots", action="store_true")

    args = parser.parse_args()

    dataset = load_data(args.data, split="train")

    logger.info(dataset.head())

    examples = dataset.Context
    definitions = dataset.Definition
    n_entries = len(dataset)
    n_lemmas = len(set(dataset['Targets'].tolist()))

    logger.info(f"Entries: {n_entries}, Lemmas: {n_lemmas}, Ratio: {n_entries / n_lemmas}")
    # glosses = df['gloss'].tolist()
    # examples = df['example'].tolist()

    # gloss_lens = [len(g.split()) - 1 for g in glosses]
    if args.model:
        tokenizer = T5Tokenizer.from_pretrained(args.model)
        encoded_tokens = tokenizer(
            examples.tolist(),
            return_tensors="pt",
            padding=True
        )["input_ids"]
        encoded_examples_no_padding = []
        for i in encoded_tokens:
            tokens = [el for el in i if el != 0]
            encoded_examples_no_padding.append(tokens)
        example_lens = [len(ex) for ex in encoded_examples_no_padding]

        encoded_tokens_definitions = tokenizer(
            definitions.tolist(),
            return_tensors="pt",
            padding=True
        )["input_ids"]
        encoded_definitions_no_padding = []
        for i in encoded_tokens_definitions:
            tokens = [el for el in i if el != 0]
            encoded_definitions_no_padding.append(tokens)
        definition_lens = [len(ex) for ex in encoded_definitions_no_padding]
    else:
        # -1 because always full stop at the end
        example_lens = [len(e.split()) - 1 for e in examples]
        definition_lens = [len(d.split()) for d in definitions]


    ax = sns.displot(example_lens, kind="hist",
                     bins=120, color='orange')

    if args.plot:
        sns.set(font_scale=1.5)
        if args.model:
            ax.set(xlabel="Example length in subwords")
            ax.set(title=f"Example lengths in subwords in {args.data}")
            ax.savefig(args.data.split("/")[0] + "_hist_subwords.png", dpi=300)
        else:
            ax.set(xlabel="Example length in words")
            ax.set(title=f"Example lengths in words in {args.data}")
            ax.savefig(args.data.split("/")[0] + "_hist.png", dpi=300)

    logger.info("Average example length: {:.2f} ± {:.2f}".format(np.mean(example_lens), np.std(example_lens)))
    logger.info("Average definition length: {:.2f} ± {:.2f}".format(np.mean(definition_lens), np.std(definition_lens)))
    logger.info("Some examples:")
    print(examples)
