#! /bin/env python3
# coding: utf-8

import logging
import torch

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def divide_chunks(data, n):
    for i in range(0, len(data), n):
        yield data[i: i + n]


def bert_embeddings(sentences, model, tokenizer, subword="mean"):
    logger.debug(sentences)
    texts = [el[0] for el in sentences]
    offsets = [el[1] for el in sentences]
    tokens = [el[2] for el in sentences]
    if torch.cuda.is_available():
        encoded_input = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                  return_offsets_mapping=True, return_attention_mask=True).to("cuda")
    else:
        encoded_input = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                  return_offsets_mapping=True, return_attention_mask=True)

    mappings = encoded_input["offset_mapping"]

    # Changing zeroes to an arbitrary large integer
    # so that they do not interfere with the real offsets:
    counter = 0
    for sent in mappings:
        for nrw, word in enumerate(sent):
            if word[0] == 0 and word[1] == 0:
                mappings[counter, nrw, :] = torch.tensor([11111, 11111])
        counter += 1

    # FIXME: proper fix for target offsets not matching BERT tokenization at all
    for nr, offset in enumerate(offsets):
        start = offset[0]
        end = offset[1]
        bert_offsets = mappings[nr]
        bert_starts = [i[0] for i in bert_offsets]
        bert_ends = [i[1] for i in bert_offsets]
        if start not in bert_starts:
            logger.debug(f"Start offset {start} not found in sentence {nr}!")
            candidate = min(bert_starts, key=lambda x: abs(x - start))
            logger.debug(candidate)
            offsets[nr][0] = candidate

        if end not in bert_ends:
            logger.debug(f"End offset {end} not found in sentence {nr}!")
            candidate = min(bert_ends, key=lambda x: abs(x - end))
            logger.debug(candidate)
            offsets[nr][1] = candidate

    if torch.cuda.is_available():
        offsets = torch.tensor(offsets).to("cuda")
    else:
        offsets = torch.tensor(offsets)
    offsets = offsets.unsqueeze(1)
    our_token_index_0 = \
        (mappings[:, :, 0] == offsets[:, :, 0]).nonzero()[:, 1]
    our_token_index_1 = \
        (mappings[:, :, 1] == offsets[:, :, 1]).nonzero()[:, 1]
    logger.debug(f"Initial subword indices: {our_token_index_0}")
    logger.debug(f"Final subword indices: {our_token_index_1}")
    # TODO: handle cases when final indices do not match BERT tokenization
    subwords = [tokenizer.convert_ids_to_tokens(el) for el in encoded_input["input_ids"]]
    logger.debug(subwords)
    #  our_subwords = [sw[i0 : i1 + 1] for sw, i0, i1
    #  in zip(subwords, our_token_index_0, our_token_index_1)]
    reconstructed_tokens = [tokenizer.decode(s[i0: i1 + 1])
                            for s, i0, i1 in zip(encoded_input["input_ids"],
                                                 our_token_index_0, our_token_index_1)]
    assert len(reconstructed_tokens) == len(tokens)
    if reconstructed_tokens != tokens:
        logger.debug(f"Original tokens: {tokens}")
        logger.debug(f"Reconstructed tokens: {reconstructed_tokens}")
    output = model(encoded_input["input_ids"])
    embeddings = output.last_hidden_state.squeeze()
    logger.debug(f"Batch embeddings shape: {embeddings.shape}")

    our_embeddings = [emb[i0:i1 + 1, :] for emb, i0, i1 in
                      zip(embeddings, our_token_index_0, our_token_index_1)]
    # Petter added: using the subword parameter
    if subword == "allbutlast":
        # Average of all but the last subword, but use the last if there is only one
        our_embeddings = [torch.mean(emb[:-1], 0) if len(emb) > 1 else emb[-1] for emb in
                          our_embeddings]
    elif subword == "first":
        our_embeddings = [emb[0] for emb in our_embeddings]
    elif subword == "last":
        our_embeddings = [emb[-1] for emb in our_embeddings]
    else:  # default to mean
        our_embeddings = [torch.mean(emb, 0) for emb in our_embeddings]  # original
    # End of Petter change
    token_embedding = torch.stack(our_embeddings)
    return token_embedding


def prepare_data(df, cur_model, cur_tokenizer, batch_size, lim=10000, distance="cosine",
                 subword="mean"):
    texts1 = df["usage1"][:lim]
    offsets1 = df["offset1"][:lim]
    tokens1 = df["token1"][:lim]
    ids1 = [val[0] for val in
            cur_tokenizer(tokens1.tolist(), add_special_tokens=False)["input_ids"]]

    raw_sentences_1 = [(text, offset, token, ident)
                       for text, offset, token, ident in zip(texts1, offsets1, tokens1, ids1)]

    texts2 = df["usage2"][:lim]
    offsets2 = df["offset2"][:lim]
    tokens2 = df["token2"][:lim]
    ids2 = [val[0] for val in
            cur_tokenizer(tokens2.tolist(), add_special_tokens=False)["input_ids"]]

    raw_sentences_2 = [(text, offset, token, ident)
                       for text, offset, token, ident in zip(texts2, offsets2, tokens2, ids2)]

    sentences_1 = []
    sentences_2 = []
    judgments = []
    # Removing sentences where target words are tokenized as UNK
    # TODO: notify if ALL pairs are like this
    unk_id = cur_tokenizer.convert_tokens_to_ids(cur_tokenizer.special_tokens_map["unk_token"])
    removed = set()
    for sent1, sent2, jud, identifier1, identifier2 in zip(
            raw_sentences_1, raw_sentences_2, df["judgment"][:lim].values,
            df["identifier1"][:lim].values, df["identifier2"][:lim].values):
        if sent1[3] != unk_id and sent2[3] != unk_id:
            sentences_1.append(sent1[:3])
            sentences_2.append(sent2[:3])
            judgments.append(jud)
        else:
            removed.add((identifier1, identifier2))

    logger.info(f"{len(raw_sentences_1) - len(sentences_1)} "
                f"pairs removed due to UNK out of total {len(raw_sentences_1)}")

    embeddings_1 = torch.zeros(len(sentences_1), cur_model.config.hidden_size)
    embeddings_2 = torch.zeros(len(sentences_2), cur_model.config.hidden_size)
    with torch.no_grad():
        logger.info(f"Inferring embeddings from first sentences...")
        chunk_counter = 0
        for chunk in divide_chunks(sentences_1, batch_size):
            cur_embeddings = bert_embeddings(chunk, cur_model, cur_tokenizer, subword=subword)
            first_row = batch_size * chunk_counter
            last_row = first_row + batch_size
            embeddings_1[first_row:last_row] = cur_embeddings
            chunk_counter += 1
            if chunk_counter % 10 == 0:
                logger.info(f"{chunk_counter} chunks processed")

        logger.info(f"Inferring embeddings from second sentences...")
        chunk_counter = 0
        for chunk in divide_chunks(sentences_2, batch_size):
            cur_embeddings = bert_embeddings(chunk, cur_model, cur_tokenizer, subword=subword)
            first_row = batch_size * chunk_counter
            last_row = first_row + batch_size
            embeddings_2[first_row:last_row] = cur_embeddings
            chunk_counter += 1
            if chunk_counter % 10 == 0:
                logger.info(f"{chunk_counter} chunks processed")
    logger.debug(embeddings_1.shape)
    logger.debug(embeddings_2.shape)
    logger.info(f"Calculating similarity/distance with {distance}")
    # Petter change:
    if distance == "manhattan":
        cos_sims = torch.nn.PairwiseDistance(p=1)(embeddings_1, embeddings_2)
    elif distance == "normalized":
        # A potential problem with the manhattan distance is the great
        # size of the numbers. We divide by the hidden dimension to 
        # mitigate this.
        cos_sims = torch.nn.PairwiseDistance(p=1)(embeddings_1,
                                                  embeddings_2) / cur_model.config.hidden_size
    elif distance == "inverted":
        # The Manhattan distance is a distance, not a similarity
        # we can invert it to have the same direction as cosine:
        cos_sims = torch.nn.PairwiseDistance(p=1)(embeddings_1,
                                                  embeddings_2) / cur_model.config.hidden_size
        cos_sims *= -1
    else:  # default to cosine:
        cos_sims = torch.nn.CosineSimilarity(dim=1)(embeddings_1, embeddings_2)
    logger.debug(f"Cosine similarities: {cos_sims}")
    logger.debug(judgments)
    cos_sims = cos_sims.unsqueeze(1)
    judgments = torch.tensor(judgments).unsqueeze(1).float()
    return cos_sims, judgments, removed


def convert_offset(offset):
    return [int(el) for el in offset.split(":")]
