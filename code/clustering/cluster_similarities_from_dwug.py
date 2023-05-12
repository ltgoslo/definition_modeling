import argparse
import itertools
import json
import logging
import os.path
import networkx as nx
from wug_modules import *


def main(args):
    if os.path.isfile(args.graphs_path):
        graph_files = [args.graphs_path]
    elif os.path.isdir(args.graphs_path):
        graph_files = sorted(
            [os.path.join(args.graphs_path, f) for f in os.listdir(args.graphs_path)])
    else:
        raise ValueError(
            "Invalid 'graphs path'. It should point to a directory of graph csv files or to a single csv file.")

    json_output = {}
    for graph_file in graph_files:
        logger.info(graph_file)
        G = nx.read_gpickle(graph_file)

        edges = get_edge_data(G)

        clusters, c2n, n2c = get_clusters(G)

        # Collect intra-cluster stats
        intra_cluster_weights = defaultdict(list)
        for c, nodes in c2n.items():
            for n1, n2 in itertools.combinations(nodes, 2):
                try:
                    edge = edges[(n1, n2)]
                except KeyError:
                    try:
                        edge = edges[(n2, n1)]
                    except KeyError:
                        continue
                w = float(edge["weight"])
                if w and not np.isnan(w):
                    intra_cluster_weights[c].append(w)

        # Collect inter-cluster stats
        inter_cluster_weights = defaultdict(list)
        for c1, c2 in itertools.combinations(c2n.keys(), 2):
            for n1, n2 in itertools.product(c2n[c1], c2n[c2]):
                try:
                    edge = edges[(n1, n2)]
                except KeyError:
                    try:
                        edge = edges[(n2, n1)]
                    except KeyError:
                        continue
                w = float(edge["weight"])
                if w and not np.isnan(w):
                    inter_cluster_weights[f"{c1}-{c2}"].append(w)

        for c in intra_cluster_weights:
            inter_cluster_weights[f"{c}-{c}"] = intra_cluster_weights[c]

        target_word = graph_file.split('/')[-1].split('.')[0].split('_')[0]
        json_output[target_word] = dict(sorted(inter_cluster_weights.items(), key=lambda x: (x[0], x[1])))

    # Store stats as dictionaries in json format
    with open(args.output_path, 'w') as outfile:
        json.dump(json_output, outfile)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--graphs_path",
        "-g",
        help="Path to a directory of graph csv files (e.g. 'dwug_en/graphs/opt') or to a single csv file.",
        required=True,
    )
    arg(
        "--output_path",
        "-o",
        help="The output path for the json file containing cluster similarities",
    )
    args = parser.parse_args()

    main(args)
