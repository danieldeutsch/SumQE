python scripts/get_summaries_jsonl.py datasets/duc_2005.json > datasets/duc2005-summaries.jsonl
python scripts/get_summaries_jsonl.py datasets/duc_2006.json > datasets/duc2006-summaries.jsonl
python scripts/get_summaries_jsonl.py datasets/duc_2007.json > datasets/duc2007-summaries.jsonl

python -m src.BERT_experiments.predict datasets/duc2005-summaries.jsonl trained_models/BERT_2005_Multi\ Task-5.npy predictions-2005.jsonl
python -m src.BERT_experiments.predict datasets/duc2006-summaries.jsonl trained_models/BERT_2006_Multi\ Task-5.npy predictions-2006.jsonl
python -m src.BERT_experiments.predict datasets/duc2007-summaries.jsonl trained_models/BERT_2007_Multi\ Task-5.npy predictions-2007.jsonl
