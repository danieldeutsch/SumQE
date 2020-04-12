import sys
import json

data = json.loads(open(sys.argv[1], 'r').read())

for doc_id, doc in data.items():
    for peer_id, peer in doc['peer_summarizers'].items():
        if len(peer['system_summary'].strip()) > 0:
            data_dict = {
                'summary': peer['system_summary'],
                'peer_id': peer_id,
                'Q1': peer['human_scores']['Q1'],
                'Q2': peer['human_scores']['Q2'],
                'Q3': peer['human_scores']['Q3'],
                'Q4': peer['human_scores']['Q4'],
                'Q5': peer['human_scores']['Q5']
            }
            print(json.dumps(data_dict))
