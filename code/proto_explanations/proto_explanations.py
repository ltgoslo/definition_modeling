#!/bin/env python3
# coding: utf-8

import argparse
import logging
import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--map", "-m", help="Path to the TSV file with sense maps", required=True)
    arg("--threshold", "-t", type=float, help="Minimum required similarity", default=0)
    args = parser.parse_args()

    data = pd.read_csv(args.map, delimiter="\t", header=0)
    logger.info(data)

    for word in data.word.unique():
        logger.info("================")
        logger.info(word)
        logger.info("================")
        word_clusters = data[data.word == word]["cluster pair (sense_period)"].unique()
        word_clusters = [eval(el) for el in word_clusters]
        cluster_periods = {}
        for cl_pair in word_clusters:
            for cl in cl_pair:
                cluster, period = cl.split("_")
                cluster = int(cluster)
                period = int(period)
                if cluster not in cluster_periods:
                    cluster_periods[cluster] = set()
                cluster_periods[cluster].add(period)
        logger.debug(cluster_periods)
        df = data[(data.word == word) & (data.edge == "LINK")]
        if not df.empty:
            synchronic = {}
            diachronic = {}
            for row in df.iterrows():
                clusters = eval(row[1]["cluster pair (sense_period)"])
                cluster1, cluster2 = clusters
                definition1 = row[1].label1
                definition2 = row[1].label2
                similarity = row[1].similarity
                if similarity < args.threshold:
                    logger.info(
                        f"Similarity too low! {clusters}\t{similarity}\t{definition1}"
                        f"\t{definition2}")
                    continue
                cluster_id1, period1 = cluster1.split("_")
                cluster_id2, period2 = cluster2.split("_")
                names = tuple(sorted([cluster_id1, cluster_id2]))
                if period1 == period2:  # same time period
                    time_period = int(period1)
                    if names not in synchronic:
                        synchronic[names] = {"periods": set(), "defs": (definition1, definition2)}
                    synchronic[names]["periods"].add(time_period)
                else:  # different time periods, diachronic relationship
                    time_period1 = int(period1)
                    time_period2 = int(period2)
                    cluster_id1 = int(cluster_id1)
                    cluster_id2 = int(cluster_id2)
                    if (cluster_id1, cluster_id2) not in diachronic:
                        diachronic[(cluster_id1, cluster_id2)]\
                            = {"periods": (time_period1, time_period2),
                               "defs": (definition1, definition2), "nature": None}
                    if len(cluster_periods[cluster_id2]) == 1 \
                            and len(cluster_periods[cluster_id1]) == 1:
                        diachronic[(cluster_id1, cluster_id2)]["nature"] = "transition"
                    elif len(cluster_periods[cluster_id2]) == 2 \
                            and len(cluster_periods[cluster_id1]) == 1:
                        diachronic[(cluster_id1, cluster_id2)]["nature"] = "merge"
                    elif len(cluster_periods[cluster_id2]) == 1 \
                            and len(cluster_periods[cluster_id1]) == 2:
                        diachronic[(cluster_id1, cluster_id2)]["nature"] = "offshoot"
                    elif len(cluster_periods[cluster_id2]) == 2 \
                            and len(cluster_periods[cluster_id1]) == 2:
                        diachronic[(cluster_id1, cluster_id2)]["nature"] = "stable"
            for link in synchronic:
                periods = synchronic[link]["periods"]
                def1 = synchronic[link]["defs"][0]
                def2 = synchronic[link]["defs"][1]
                if def1 == def2:
                    logger.info(f"It seems that two senses of '{word}' at time period(s) {periods} "
                                f"are identical and should probably be clustered together:")
                else:
                    logger.info(f"It seems that two senses of '{word}' at time period(s) {periods} "
                                f"are very similar and should probably be clustered together. "
                                f"May be, they are sub-senses of the same super-sense:")
                logger.info(f"Sense cluster {link[0]}: {def1}")
                logger.info(f"Sense cluster {link[1]}: {def2}")
            for link in diachronic:
                periods = diachronic[link]["periods"]
                def1 = diachronic[link]["defs"][0]
                def2 = diachronic[link]["defs"][1]
                if diachronic[link]["nature"] == "stable":
                    continue
                elif diachronic[link]["nature"] == "transition":
                    logger.info(f"DIA It seems that one sense of '{word}' from time period "
                                f"{periods[0]} died but gave birth to a novel sense "
                                f"in time period {periods[1]}:")
                    logger.info(f"Sense cluster {link[0]} in period {periods[0]}: {def1}")
                    logger.info(f"Sense cluster {link[1]} in period {periods[1]}: {def2}")
                elif diachronic[link]["nature"] == "merge":
                    logger.info(f"DIA It seems that one sense of '{word}' from time period "
                                f"{periods[0]} merged into a wider sense:")
                    logger.info(f"Sense cluster {link[0]} in period {periods[0]}: {def1}")
                    logger.info(f"Sense cluster {link[1]} in both periods: {def2}")
                elif diachronic[link]["nature"] == "offshoot":
                    logger.info(f"DIA It seems that a novel sense of '{word}' in time period "
                                f"{periods[1]} is probably an offshoot of a wider sense:")
                    logger.info(f"Sense cluster {link[1]} in period {periods[1]}: {def2}")
                    logger.info(f"Sense cluster {link[0]} in both periods: {def1}")
        else:
            logger.info(f"No interesting links found for the word '{word}'.")
