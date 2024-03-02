import json
import gzip

def gen_prompt(input_data, num_retrieved_docs):
    prompt = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."
    prompt += "\n\n"
    
    for doc_id, document in enumerate(input_data['ctxs'][:num_retrieved_docs]):
        prompt += "Document [" + str(doc_id+1) + "](Title: " + document['title'] + ") " + document['text'] + '\n'
    
    prompt += "\n" 
    prompt += "Question: " + input_data['question']
    prompt += "\n" + "Answer:"

    return prompt


num_retrieved_docs = 5
data_path = '../download/nq-open-10_total_documents_gold_at_0.jsonl.gz'
output_path = '../prompts/rag_nq_' + str(num_retrieved_docs) + '.json'

prompt_list = []

with gzip.open(data_path, 'rb') as f:
    for line in f:
        data = json.loads(line)
        prompt_list.append(gen_prompt(data, num_retrieved_docs))

with open(output_path, 'w') as f:
    json.dump(prompt_list, f)