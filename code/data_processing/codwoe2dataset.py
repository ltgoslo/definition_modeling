#! /bin/env/python3

import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])

ft = df[["word","gloss","example"]]

ft.to_csv(sys.argv[1].replace(".csv", ".tsv"), sep="\t", index=False)
