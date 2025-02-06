#! /bin/env/python3

import csv
from smart_open import open
import sys
import json
import pandas as pd

prompts = [
". Что такое <TRG>?",  # 0
"Hva betyr <TRG>?",  # 1
"Was ist die Definition von <TRG>?",  # 2
". Mikä on <TRG>?", # 3
]

prompt = prompts[int(sys.argv[3])]

dataset = []

with open(sys.argv[1]) as f:
    for line in f:
        data = json.loads(line.strip())
        definition = data["definition"]
        example = "".join([data["example"], prompt.replace("<TRG>", data["target"])])
        dataset.append([example, definition])

df = pd.DataFrame(dataset)

df.columns = ["example", "definition"]

df.to_csv(sys.argv[2], sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_NONE)
