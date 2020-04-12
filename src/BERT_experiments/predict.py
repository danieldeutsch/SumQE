import argparse
import json
import numpy as np
import os
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from typing import List

from src.BERT_experiments.Bert_model import compile_bert
from src.vectorizer import BERTVectorizer


def load_summaries(input_file: str) -> List[str]:
    summaries = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            summaries.append(data['summary'])
    return summaries


def main(args):
    summaries = load_summaries(args.input_summaries_jsonl)

    input_dict = {
        'word_inputs': [],
        'pos_inputs': [],
        'seg_inputs': []
    }

    vectorizer = BERTVectorizer()
    for summary in tqdm(summaries, desc='Tokenizing summaries'):
        summary_sentences = sent_tokenize(summary)

        tok_ids = vectorizer.vectorize_inputs(sequence=summary_sentences[0], i=0)

        for i, sentence in enumerate(summary_sentences[1:]):
            sentence_tok = vectorizer.vectorize_inputs(sequence=sentence, i=i + 1)
            tok_ids = tok_ids + sentence_tok

        inputs = vectorizer.transform_to_inputs(tok_ids)

        input_dict['word_inputs'].append(inputs[0, 0])
        input_dict['pos_inputs'].append(inputs[1, 0])
        input_dict['seg_inputs'].append(inputs[2, 0])

    for name, input_values in input_dict.items():
        input_dict[name] = np.array(input_values)

    # Load the model and run prediction
    print(f'Loading model from {args.model_file}')
    weights = np.load(args.model_file, allow_pickle=True)
    model = compile_bert(shape=(512, 512), dropout_rate=0.1, lr=2e-5, mode='Multi Task-5')
    model.set_weights(weights)

    print('Predicting scores')
    prediction = model.predict(input_dict, batch_size=1)

    print(f'Saving scores to {args.output_json}')
    dirname = os.path.dirname(args.output_json)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.output_json, 'w') as out:
        out.write(json.dumps(prediction.tolist()))


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_summaries_jsonl')
    argp.add_argument('model_file')
    argp.add_argument('output_json')
    args = argp.parse_args()
    main(args)
