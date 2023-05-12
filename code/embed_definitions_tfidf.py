import argparse
import csv
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def clean_generated_definition(definition):
    clean_definition = definition.strip()
    if clean_definition[-1] == ".":
        clean_definition = clean_definition[:-1]
    return clean_definition



def main(args):

    definitions_df = pd.read_csv(
        args.input_path, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
    )
    if args.debug_instances:
        definitions_df = definitions_df[:args.debug_instances]

    index2id = []
    for entry_id in definitions_df[args.key_to_entry_id]:
        index2id.append(entry_id)

    definition_corpus = []
    for definition in definitions_df.Definitions:
        definition_corpus.append(clean_generated_definition(definition))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)
    tfidf_output = vectorizer.fit_transform(definition_corpus)
    print(tfidf_output.shape)
    embeddings = {}
    for i in range(tfidf_output.shape[0]):
        target_id = index2id[i]
        embeddings[target_id] = tfidf_output[i].toarray().squeeze()

    if not args.output_path:
        args.output_path = os.path.join(args.input_path.split(".")[0])
    np.savez_compressed(args.output_path, **embeddings)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--input_path",
        "-i",
        help="Path to the tsv file containing generated definitions",
        required=True,
    )
    arg(
        "--key_to_entry_id",
        "-k",
        default="id",
        help="The name of the entry id column of the input dataframe. It varies across datasets ('Sense', 'id', 'Targets')",
    )
    arg(
        "--output_path",
        "-o",
        help="Where to save the definition embeddings",
        required=False
    )
    arg(
        "--debug_instances",
        "-d",
        type=int,
        default=None,
        help="The number of definitions to embed. If none, all definitions are embedded.",
    )

    # arg("--maxl", "-ml", type=int, help="Max source length", default=256)
    args = parser.parse_args()

    main(args)
