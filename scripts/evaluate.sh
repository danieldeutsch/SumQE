echo '2005'
python -m src.BERT_experiments.evaluate datasets/duc2005-summaries.jsonl predictions-2005.jsonl

echo '2006'
python -m src.BERT_experiments.evaluate datasets/duc2006-summaries.jsonl predictions-2006.jsonl

echo '2007'
python -m src.BERT_experiments.evaluate datasets/duc2007-summaries.jsonl predictions-2007.jsonl
