#!/bin/env python3
# coding: utf-8


import pandas as pd
import sys
from random import choice
import json

senses = pd.read_csv(sys.argv[1], delimiter="\t")
print(senses)

bits = [0, 1]

swap = [choice(bits) for el in range(len(senses))]

print(swap)

with open(sys.argv[2], "w") as f:
    f.write(json.dumps(swap))

for nr, val in enumerate(swap):
    if val == 1:
        senses.loc[nr, ["from_def", "from_use"]] = senses.loc[nr, ["from_use", "from_def"]].values

senses = senses.rename(columns={"from_def": "System1", "from_use": "System2"})

senses["Judgments"] = ""

print(senses)

senses.to_csv(sys.argv[1].replace("annotation", "annotation_randomized"), sep="\t", index=False)

with pd.ExcelWriter(sys.argv[1].replace(
        "annotation", "annotation_randomized").replace(".tsv", ".ods")) as writer:
    senses.to_excel(writer)
