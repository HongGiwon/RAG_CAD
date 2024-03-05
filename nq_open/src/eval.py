import json
import gzip

data_path = '../download/nq-open-10_total_documents_gold_at_0.jsonl.gz'

output_paths = [
    '../outputs/rag_nq_5_Llama-2-7b-chat-hf_short.json',
    '../outputs/rag_nq_5_Llama-2-13b-chat-hf_short.json',
    '../outputs/rag_nq_5_cad_Llama-2-7b-chat-hf_short.json',
    '../outputs/rag_nq_5_cad_Llama-2-13b-chat-hf_short.json'
]

with gzip.open(data_path, 'rb') as f:
        for line in f:
            data = json.loads(line)
            short_answer_list.append(data['answers'])

for output_path in output_paths:
    short_answer_list = []
    
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
