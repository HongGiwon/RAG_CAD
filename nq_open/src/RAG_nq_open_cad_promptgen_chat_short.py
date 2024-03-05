import json
import gzip

def gen_prompt_single(input_data, target_doc_id):
    prompt = []
    prompt.append(
        {
            "role" : "system",
            "content": "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Provide the answer in 5 words or less without any explanation."
        }
    )

    # adding a dummy example for zero-shot
    if target_doc_id == -1: ##blank
        user_input = "Question: Question\n" + "Answer:"
    else:
        user_input = "Document [1](Title: Title) Text\n\nQuestion: Question\n" + "Answer:"
    prompt.append(
        {
            "role" : "user",
            "content": user_input
        }
    )
    prompt.append(
        {
            "role" : "assistant",
            "content": "March 2018"
        }
    )

    if target_doc_id == -1: ##blank
        user_input = ""
    else:
        document = input_data['ctxs'][target_doc_id]
        user_input = "Document [" + str(1) + "](Title: " + document['title'] + ") " + document['text'] + '\n\n'

    prompt.append(
        {
            "role" : "user",
            "content": user_input + "Question: " + input_data['question'] + "\n" + "Answer:"
        }
    )
    return prompt


num_retrieved_docs = 5
ans_pos = 4
data_path = '../download/nq-open-10_total_documents_gold_at_' + str(ans_pos) + '.jsonl.gz'
output_path = '../prompts/rag_nq_' + str(num_retrieved_docs) + '_cad_chat_short_' + str(ans_pos) + '.json'

prompt_list = []

with gzip.open(data_path, 'rb') as f:
    for line in f:
        data = json.loads(line)
        prompt_cad_list = []
        for i in range(num_retrieved_docs):
            if data['ctxs'][i]['isgold']:
                score = "gold"
            else:
                score = data['ctxs'][i]['score']
            prompt_cad_list.append((gen_prompt_single(data, i), score))
        prompt_cad_list.append((gen_prompt_single(data, -1), "empty"))
        prompt_list.append(prompt_cad_list)

with open(output_path, 'w') as f:
    json.dump(prompt_list, f)
