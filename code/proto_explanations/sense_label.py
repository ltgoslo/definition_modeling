#!/bin/env python3
# coding: utf-8

import argparse
import logging
import csv
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from embed_definitions import _mean_pooling
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pylab as plot
from random import sample


def vizualize(vectors, classes, seq_classes, labels, protos, lemma, method="PCA"):
    if method == "PCA":
        reduction = PCA(n_components=2, random_state=0)
    else:
        perplexity = int(len(seq_classes) ** 0.5)
        reduction = TSNE(
            n_components=2,
            perplexity=perplexity,
            metric="cosine",
            n_iter=500,
            init="pca",
        )

    matrix = np.vstack(vectors)
    y = reduction.fit_transform(matrix)

    colors = plot.cm.rainbow(np.linspace(0, 1, len(classes)))

    class2color = [colors[classes.index(w)] for w in seq_classes]

    xpositions = y[:, 0]
    ypositions = y[:, 1]
    seen = set()

    plot.clf()

    for color, class_label, mark, x, y in zip(
        class2color, seq_classes, protos, xpositions, ypositions
    ):
        plot.scatter(
            x,
            y,
            200 if mark == 1 else 40,
            marker="*",
            alpha=0.8,
            color=color,
            label=labels[class_label]
            if mark != 1
            and class_label not in seen
            and labels[class_label] != "Too small sense"
            else "",
        )
        if mark != 1:
            seen.add(class_label)

    plot.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    plot.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    plot.legend(loc="best")
    plot.title(lemma)

    # plot.show()
    plot.savefig(
        f"{lemma}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plot.close()
    plot.clf()
    return plot


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--data",
        "-d",
        help="Path to the tsv file with definitions, usages and clusters",
        required=True,
    )
    arg(
        "--model",
        "-m",
        help="Path to a Huggingface model",
        default="sentence-transformers/all-distilroberta-v1",
    )  # for other languages: sentence-transformers/distiluse-base-multilingual-cased-v1
    arg("--bsize", "-b", type=int, help="Batch size", default=4)
    arg(
        "--mode",
        type=str,
        choices=["definition", "usage"],
        help="Find most prototypical definition or most prototypical usage?",
        default="definition",
    )
    arg(
        "--save",
        "-s",
        type=str,
        choices=["plot", "text"],
        help="Save plots or usages and definitions?",
        default="plot",
    )
    arg("--examples", "-e", type=bool, help="Do we need examples?", default=False)
    arg(
        "--output",
        "-o",
        help="Where to save a file with cluster labels?",
        default="sense_labels.tsv"
      )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval_metrics().to(device)

    dataset = pd.read_csv(args.data, delimiter="\t",quoting=csv.QUOTE_NONE)
    lemmas = sorted(set(dataset.word.values))

    labels_df = pd.DataFrame(
        {"Targets": [], "Examples": [], "Definitions": [], "Clusters": []}
    )
    nr_examples = 5

    for word in tqdm(lemmas):
        logger.debug(f"Processing {word}...")
        df = dataset[(dataset.word == word) & (dataset.cluster != -1)]
        senses = sorted(set(df.cluster.values))

        sense_matrix = []
        proto_definitions = []
        markers = []
        sense_labels = []
        for sense in senses:
            definitions = df[(df.cluster == sense)]["Definitions"].tolist()
            usages = df[(df.cluster == sense)]["Real_Contexts"].tolist()
            assert len(definitions) == len(usages)
            if args.mode == "usage":
                representations = usages
            else:
                representations = definitions
            proto_markers = [0 for el in range(len(representations))]

            inputs = tokenizer(
                representations,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            embeddings = np.zeros((len(representations), model.config.hidden_size))
            token_dataset = torch.utils.data.TensorDataset(
                inputs.input_ids,
                inputs.attention_mask,
                torch.tensor(np.arange(len(representations))),
            )
            dataloader = torch.utils.data.DataLoader(
                token_dataset, batch_size=args.bsize, shuffle=False
            )
            for _inputs, att_masks, target_indices in dataloader:
                with torch.no_grad():
                    model_output = model(
                        input_ids=_inputs.to(device),
                        attention_mask=att_masks.to(device),
                        output_hidden_states=True,
                    )

                sentence_embeddings = _mean_pooling(model_output, att_masks.to(device))
                sentence_embeddings = torch.nn.functional.normalize(
                    sentence_embeddings, dim=1).to("cpu")
                embeddings[
                    target_indices[0] : target_indices[-1] + 1, :
                ] = sentence_embeddings
            if len(representations) < 3:
                logger.debug(f"Sense {sense}:")
                logger.debug(
                    f"Too few usages/definitions for this sense: {len(representations)}. "
                    f"At least 3 required"
                )
                proto_definitions.append("Too small sense")
                prototype_definition = "Too few examples to generate a proper definition!"
            else:
                logger.debug(f"Sense {sense}:")
                prototype_embedding = np.mean(embeddings, axis=0)
                sims = np.dot(embeddings, prototype_embedding)
                proto_index = np.argmax(sims)
                if args.mode == "usage":
                    prototype_usage = usages[proto_index]
                    logger.info(prototype_usage)
                prototype_definition = definitions[proto_index]
                logger.debug(prototype_definition)
                proto_definitions.append(prototype_definition)
                proto_markers[proto_index] = 1
            if args.save == "text":
                if args.examples:
                    sample_size = (
                        nr_examples if len(usages) >= nr_examples else len(usages)
                    )
                    cur_examples = sample(usages, k=sample_size)
                    cur_examples = "\n".join(cur_examples)
                else:
                    cur_examples = ""
            labels_df.loc[len(labels_df)] = [
                word,
                cur_examples,
                prototype_definition,
                int(sense),
                    ]
            sense_matrix.append(embeddings)
            sense_labels += [sense for el in range(len(definitions))]
            markers += proto_markers

        if args.save == "plot":
            image = vizualize(
                sense_matrix, senses, sense_labels, proto_definitions, markers, word
            )

    if args.save == "text":
        labels_df.to_csv(args.output, sep="\t", index=False, quoting=csv.QUOTE_NONE)
