import argparse
import csv
import itertools
import logging
from collections import defaultdict
import evaluate
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import pairwise
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.hidden_states[-1]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_definition_embeddings(definitions, tokenizer, model, device):  #, transform="normalise_standardise"):
    # if transform not in ["standardise", "normalise", "normalise_standardise", None]:
    #     raise ValueError()
    inputs = tokenizer(
        definitions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        model_output = model(**inputs.to(device), output_hidden_states=True)
        embeddings = _mean_pooling(model_output, inputs.attention_mask)
        # if transform == "standardise":
        #     embeddings = StandardScaler().fit_transform(embeddings.to("cpu"))
        # elif transform == "normalise":
        #     embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).to("cpu")
        # elif transform == "normalise_standardise":
        #     embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).to("cpu")
        #     embeddings = StandardScaler().fit_transform(embeddings)
        # else:
        #
        embeddings = embeddings.to("cpu")
    return embeddings


def main(args):
    definitions_df = pd.read_csv(
        args.data_path, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
    )
    if args.debug_instances:
        definitions_df = definitions_df[:args.debug_instances]

    definitions_df["Definitions"] = definitions_df["Definitions"].fillna("")

    word2ids = defaultdict(list)
    word2defs = defaultdict(list)
    for _, row in definitions_df.iterrows():
        word2ids[row["word"]].append(row["id"])
        word2defs[row["word"]].append(row["Definitions"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    roberta = AutoModel.from_pretrained("roberta-large").eval().to(device)
    sent_roberta_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
    sent_roberta = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1").eval().to(device)
    # AutoModelWithLMHead.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased-v1")

    # bertscore = evaluate.load("bertscore", device=device)
    sacrebleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")

    all_embeds_roberta = []
    all_embeds_sent_roberta = []
    word2indices = {}
    for word in tqdm(sorted(word2ids.keys())):
        start_idx = len(all_embeds_roberta)
        all_embeds_roberta.extend(
            get_definition_embeddings(word2defs[word], roberta_tokenizer, roberta, device).tolist())
        all_embeds_sent_roberta.extend(
            get_definition_embeddings(word2defs[word], sent_roberta_tokenizer, sent_roberta, device).tolist())
        word2indices[word] = (start_idx, len(all_embeds_roberta))
        assert len(all_embeds_roberta) == len(all_embeds_sent_roberta)

    all_embeds_roberta = StandardScaler().fit_transform(np.array(all_embeds_roberta))
    all_embeds_sent_roberta = StandardScaler().fit_transform(np.array(all_embeds_sent_roberta))
    word2roberta_embeds = {}
    word2sent_roberta_embeds = {}
    for word in word2indices:
        start_idx, end_idx = word2indices[word]
        word2roberta_embeds[word] = all_embeds_roberta[start_idx:end_idx]
        word2sent_roberta_embeds[word] = all_embeds_sent_roberta[start_idx:end_idx]

    similarity_judgements = []
    for word in tqdm(sorted(word2ids.keys())):

        pairwise_similarities_roberta = pairwise.cosine_similarity(word2roberta_embeds[word])
        pairwise_similarities_sent_roberta = pairwise.cosine_similarity(word2sent_roberta_embeds[word])

        idx_pairs = set(itertools.combinations(list(np.arange(len(word2ids[word]))), 2))
        for i, j in idx_pairs:
            cosine_sim_roberta = pairwise_similarities_roberta[i][j]
            cosine_sim_sent_roberta = pairwise_similarities_sent_roberta[i][j]
            bleu_score = sacrebleu.compute(
                predictions=[word2defs[word][i]], references=[word2defs[word][j]]
            )["score"]
            # bert_f1 = bertscore.compute(
            #     predictions=[word2defs[word][i]], references=[word2defs[word][j]], lang="en"
            # )["f1"]
            meteor_score = meteor.compute(
                predictions=[word2defs[word][i]], references=[word2defs[word][j]], alpha=0.5
            )["meteor"]

            similarity_judgements.append((
                word,
                word2ids[word][i],
                word2ids[word][j],
                cosine_sim_roberta,
                cosine_sim_sent_roberta,
                bleu_score,
                # bert_f1,
                meteor_score
            ))

    df = pd.DataFrame(
        similarity_judgements,
        columns=["word", "id1", "id2", "cosine_roberta", "cosine_sent_roberta", "bleu", "meteor"]
        # columns=["word", "id1", "id2", "cosine_roberta", "cosine_sent_roberta", "bleu", "bertf1", "meteor"]
    )

    if not args.output_path:
        args.output_path = args.data_path.split(".")[0] + ".sim.tsv"
    df.to_csv(args.output_path, sep="\t", quoting=csv.QUOTE_NONE, index=False)


if __name__ == '__main__':

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--data_path",
        "-d",
        help="Path to the generated definitions",
        required=True,
    )
    arg(
        "--output_path",
        "-o",
        help="Where to save the similarity scores",
        required=False
    )
    arg(
        "--debug_instances",
        "-db",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    main(args)
