# Interpretable Word Sense Representations via Definition Generation

This repository accompanies the paper [`Interpretable Word Sense Representations via Definition Generation: The Case of Semantic Change Analysis`](https://aclanthology.org/2023.acl-long.176/) (ACL'2023) by Mario Giulianelli, Iris Luden, Raquel Fernández and Andrey Kutuzov.

The project is a collaboration between the [Dialogue Modelling Group](https://dmg-illc.github.io/dmg/) at the University of Amsterdam 
and the [Language Technology Group](https://www.mn.uio.no/ifi/english/research/groups/ltg/) at the University of Oslo.

## Definition generation models for English:
- [FLAN-T5-Definition Base (250M parameters)](https://huggingface.co/ltg/flan-t5-definition-en-base)
- [FLAN-T5-Definition Large (780M parameters)](https://huggingface.co/ltg/flan-t5-definition-en-large)
- [FLAN-T5-Definition XL (3B parameters)](https://huggingface.co/ltg/flan-t5-definition-en-xl)

## Citation
```
@inproceedings{giulianelli-etal-2023-interpretable,
    title = "Interpretable Word Sense Representations via Definition Generation: The Case of Semantic Change Analysis",
    author = "Giulianelli, Mario  and
      Luden, Iris  and
      Fernandez, Raquel  and
      Kutuzov, Andrey",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.176",
    doi = "10.18653/v1/2023.acl-long.176",
    pages = "3130--3148",
    abstract = "We propose using automatically generated natural language definitions of contextualised word usages as interpretable word and word sense representations. Given a collection of usage examples for a target word, and the corresponding data-driven usage clusters (i.e., word senses), a definition is generated for each usage with a specialised Flan-T5 language model, and the most prototypical definition in a usage cluster is chosen as the sense label. We demonstrate how the resulting sense labels can make existing approaches to semantic change analysis more interpretable, and how they can allow users {---} historical linguists, lexicographers, or social scientists {---} to explore and intuitively explain diachronic trajectories of word meaning. Semantic change analysis is only one of many possible applications of the {`}definitions as representations{'} paradigm. Beyond being human-readable, contextualised definitions also outperform token or usage sentence embeddings in word-in-context semantic similarity judgements, making them a new promising type of lexical representation for NLP.",
}
```
