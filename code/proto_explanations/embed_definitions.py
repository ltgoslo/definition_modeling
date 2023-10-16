import argparse
import csv
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.hidden_states[-1]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def main(arguments):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    model = (
        AutoModel.from_pretrained(arguments.model)
        .eval()
        .to(device)
    )
    # AutoModelWithLMHead.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased-v1")

    definitions_df = pd.read_csv(
        arguments.input_path, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8"
    )
    if arguments.debug_instances:
        definitions_df = definitions_df[:arguments.debug_instances]

    inputs = tokenizer(
        definitions_df.Definitions.fillna('').tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    logger.info("Tokenizing finished.")
    index2id = []
    for entry_id in definitions_df[arguments.key_to_entry_id]:
        index2id.append(entry_id)

    dataset = torch.utils.data.TensorDataset(
        inputs.input_ids.to(device),
        inputs.attention_mask.to(device),
        torch.tensor(np.arange(len(index2id))).to(device),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=arguments.bsize, shuffle=False
    )
    logger.info(f"Generating definitions embeddings with batch size {arguments.bsize}...")

    embeddings = {}
    for _inputs, att_masks, target_indices in tqdm(dataloader):
        with torch.no_grad():
            model_output = model(input_ids=_inputs.to(device), attention_mask=att_masks.to(device),
                                 output_hidden_states=True)

        sentence_embeddings = _mean_pooling(model_output, att_masks)
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1).to("cpu")
        for i, target_index in enumerate(target_indices):
            target_id = index2id[target_index.item()]
            embeddings[target_id] = sentence_embeddings[i]

    if not arguments.output_path:
        arguments.output_path = arguments.input_path.split(".")[0]
    np.savez_compressed(arguments.output_path, **embeddings)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--model",
        "-m",
        help="Path to a Huggingface model",
        default="sentence-transformers/all-distilroberta-v1",
    )  # for other languages: sentence-transformers/distiluse-base-multilingual-cased-v1
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
        help="The name of the entry id column of the input dataframe. It varies across datasets "
             "('Sense', 'id', 'Targets')",
    )
    arg("--bsize", "-b", type=int, help="Batch size", default=4)
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
