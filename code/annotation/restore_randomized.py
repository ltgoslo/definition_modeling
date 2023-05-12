#!/bin/env python3
# coding: utf-8

import pandas as pd
import sys
import json

annotation = pd.read_csv(sys.argv[1], delimiter="\t")
print(annotation)

mapping = {1: 2, 11: 22, 2: 1, 22: 11}

with open(sys.argv[2], "r") as f:
    swap = json.load(f)

for nr, val in enumerate(swap):
    if val == 1:
        annotation.loc[nr, ["System1", "System2"]] = \
            annotation.loc[nr, ["System2", "System1"]].values
        if annotation["Judgments"].loc[nr] in mapping:
            annotation["Judgments"].loc[nr] = mapping[annotation["Judgments"].loc[nr]]

print(annotation)

annotation.to_csv(sys.argv[1].replace(".tsv", "_restored.tsv"), sep="\t", index=False)
