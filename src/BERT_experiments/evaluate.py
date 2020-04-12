import argparse
import json
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau


def load_gold_labels(input_file: str):
    gold = []
    with open(input_file, 'r') as f:
        for line in f:
            g = []
            data = json.loads(line)
            g.append(data['Q1'])
            g.append(data['Q2'])
            g.append(data['Q3'])
            g.append(data['Q4'])
            g.append(data['Q5'])
            gold.append((data['peer_id'], g))
    return gold


def main(args):
    golds = load_gold_labels(args.gold_jsonl)
    preds = json.loads(open(args.prediction_json, 'r').read())

    for i in range(5):
        peer_id_to_golds = defaultdict(list)
        peer_id_to_preds = defaultdict(list)

        for (peer_id, gold), pred in zip(golds, preds):
            peer_id_to_golds[peer_id].append(gold[i])
            peer_id_to_preds[peer_id].append(pred[i])

        values1, values2 = [], []
        for peer_id in peer_id_to_golds.keys():
            values1.append(np.mean(peer_id_to_golds[peer_id]))
            values2.append(np.mean(peer_id_to_preds[peer_id]))

        print(f'Q{i + 1}')
        print('Spearman', spearmanr(values1, values2)[0])
        print('Kendall', kendalltau(values1, values2)[0])
        print('Pearson', pearsonr(values1, values2)[0])
        print()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('gold_jsonl')
    argp.add_argument('prediction_json')
    args = argp.parse_args()
    main(args)
