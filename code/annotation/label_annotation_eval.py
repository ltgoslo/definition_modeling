#!/bin/env python3
# coding: utf-8

import argparse
import logging
from os import walk, path
from statistics import multimode
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import krippendorff

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--data", "-d", help="Path to the directory with annotation TSVs",
        required=True)
    args = parser.parse_args()

    dataframes = []
    annotators = []
    for root, dirs, files in walk(args.data):
        for f in files:
            if f.endswith("_restored.tsv"):
                logger.info(f)
                annotators.append(f.replace("_restored.tsv", ""))
                annotation = pd.read_csv(path.join(root, f), header=0, delimiter="\t")
                logger.debug(annotation)
                dataframes.append(annotation)

    logger.info(f"We have {len(dataframes)} annotation files")

    judgments = []
    for nr in range(len(dataframes[0])):
        cur_judgments = [df["Judgments"].iloc[nr] for df in dataframes]
        judgments.append(cur_judgments)

    logger.info("Calculating inter-rater agreement...")
    reliability_matrix = np.zeros((len(dataframes), len(judgments)))
    for nr, instance in enumerate(judgments):
        reliability_matrix[:, nr] = instance

    agreement = krippendorff.alpha(reliability_data=reliability_matrix,
                                   level_of_measurement="nominal")
    logger.info(f"Nominal Krippendorff's alpha is {agreement:0.4f}")
    reliability_matrix[reliability_matrix == 1] = 11
    reliability_matrix[reliability_matrix == 2] = 22
    agreement = krippendorff.alpha(reliability_data=reliability_matrix,
                                   level_of_measurement="nominal")
    logger.info(f"Nominal Krippendorff's alpha on simplified data is {agreement:0.4f}")


    logger.info("Looking for cases of radical disagreement...")
    print("Nr\tTarget\tSystem1\tSystem2\t" + "\t".join(annotators))
    counter_both = 0
    for nr, j in enumerate(judgments):
        if 0 in j and 3 in j:
            logger.debug(f"0 and 3 occurred simultaneously at row {nr}!")
            data = dataframes[0][["Targets", "System1", "System2"]].iloc[nr].values
            print(str(nr) + "\t" + "\t".join(data) + "\t" + "\t".join([str(a) for a in j]))
            logger.debug(j)
            counter_both += 1

    counter_bad = 0
    for nr, j in enumerate(judgments):
        if 11 in j and 22 in j:
            logger.debug(f"11 and 22 occurred simultaneously at row {nr}!")
            data = dataframes[0][["Targets", "System1", "System2"]].iloc[nr].values
            print(str(nr) + "\t" + "\t".join(data) + "\t" + "\t".join([str(a) for a in j]))
            logger.debug(j)
            counter_bad += 1

    logger.info(f"{counter_both} cases when 0 and 3 occurred simultaneously")
    logger.info(f"{counter_bad} cases when 11 and 22 occurred simultaneously")

    mapping_success_1 = {0: 0, 1: 1, 11: 1, 2: 1, 22: 0, 3: 1}
    mapping_success_2 = {0: 0, 1: 1, 11: 0, 2: 1, 22: 1, 3: 1}
    mapping_1_better_2 = {0: 2, 1: 1, 11: 1, 2: 0, 22: 0, 3: 2}
    mapping_both = {0: 0, 1: 1, 11: 1, 2: 1, 22: 1, 3: 2}

    judgments1 = []  # How often System 1 yielded good enough results?
    judgments2 = []  # How often System 2 yielded good enough results?
    judgments1better = []  # How often System 1 was better than System 2?
    bothbad = []

    ties1 = 0
    ties2 = 0

    for j in judgments:
        map1 = [mapping_success_1[el] for el in j]
        voting = multimode(map1)
        if len(voting) > 1:
            ties1 += 1
        judgments1.append(max(voting))
        map2 = [mapping_success_2[el] for el in j]
        voting = multimode(map2)
        if len(voting) > 1:
            ties2 += 1
        judgments2.append(max(voting))
        map3 = [mapping_1_better_2[el] for el in j]
        judgments1better.append(max(multimode(map3)))
        map4 = [mapping_both[el] for el in j]
        bothbad.append(max(multimode(map4)))

    binary_better = [el for el in judgments1better if el != 2]
    logger.info(f"Ties in voting for System 1: {ties1}")
    logger.info(f"Ties in voting for System 2: {ties2}")
    logger.debug(judgments1)
    logger.debug(judgments2)
    logger.debug(judgments1better)
    logger.debug(bothbad)

    system1good = judgments1.count(1) / len(judgments1)
    system2good = judgments2.count(1) / len(judgments2)
    system1better = judgments1better.count(1) / len(judgments1better)
    system1worse = judgments1better.count(0) / len(judgments1better)
    system1onpar = judgments1better.count(2) / len(judgments1better)

    bothsystemsbad = bothbad.count(0) / len(bothbad)
    somethinggood = bothbad.count(1) / len(bothbad)
    bothgood = bothbad.count(2) / len(bothbad)

    logger.info(f"System 1 was good enough with the probability of {system1good:0.2f}")
    logger.info(f"System 2 was good enough with the probability of {system2good:0.2f}")
    diff_good_enough = stats.ttest_ind(judgments1, judgments2)
    logger.info(f"Significance of the good enough difference (p-value): {diff_good_enough[1]:0.4f}")
    logger.info(f"System 1 was better with the probability of {system1better:0.2f}")
    logger.info(f"System 1 was worse with the probability of {system1worse:0.2f}")
    diff_better = stats.ttest_ind(binary_better, [0 if el == 1 else 1 for el in binary_better])
    logger.info(f"Significance of System 1 better difference (p-value): {diff_better[1]:0.4f}")
    logger.info(f"System 1 was equal with System 2 with the probability of {system1onpar:0.2f}")

    plt.bar(["Definitions better", "Usages better", "Equal"],
            [system1better, system1worse, system1onpar])
    plt.ylabel("Probability")
    plt.title("Sense labels from definitions and usages")
    #plt.show()
    plt.savefig("definitions_usages_better.png", dpi=300, bbox_inches="tight")
    plt.clf()

    logger.info(f"Both systems were bad with the probability of {bothsystemsbad:0.2f}")
    logger.info(f"One system was good with the probability of {somethinggood:0.2f}")
    logger.info(f"Both systems were good with the probability of {bothgood:0.2f}")

    plt.bar(["Both labels bad", "One of the labels is good", "Both labels good"],
            [bothsystemsbad, somethinggood, bothgood])
    plt.ylabel("Probability")
    plt.title("General quality of two sense labels (from definitions and usages)")
    plt.savefig("general_quality_labels.png", dpi=300, bbox_inches="tight")
    #plt.show()
