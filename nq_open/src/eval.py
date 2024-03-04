import json
import gzip

data_path = '../download/nq-open-10_total_documents_gold_at_0.jsonl.gz'
output_path = '../outputs/rag_nq_10_cad_Llama-2-13b-chat-hf.json'

short_answer_list = []

with gzip.open(data_path, 'rb') as f:
    for line in f:
        data = json.loads(line)
        short_answer_list.append(data['answers'])

with open(output_path, 'r') as f:
    output = json.load(f)

em_include = 0
for pred, golds in zip(output,short_answer_list):
    ans_flag = False
    for gold in golds:
        if gold in pred:
            ans_flag = True
            break
    em_include += int(ans_flag)

print(em_include/len(output))