import json
import gzip

data_path = '../download/nq-open-10_total_documents_gold_at_0.jsonl.gz'

output_paths = [
    # # '../outputs/rag_nq_5_Llama-2-7b-chat-hf_short.json',
    # # '../outputs/rag_nq_5_Llama-2-7b-chat-hf_short_4.json',
    # # '../outputs/rag_nq_5_Llama-2-7b-chat-hf_short_9.json',
    # # # '../outputs/rag_nq_5_Llama-2-13b-chat-hf_short.json',
    # # # '../outputs/rag_nq_5_Llama-2-13b-chat-hf_short_4.json',
    # # # '../outputs/rag_nq_5_Llama-2-13b-chat-hf_short_9.json',
    # # '../outputs/rag_nq_5_cad_Llama-2-7b-chat-hf_short.json',
    # # '../outputs/rag_nq_5_cad_Llama-2-7b-chat-hf_short_4.json',
    # # '../outputs/rag_nq_5_cad_Llama-2-7b-chat-hf_short_9.json',
    # # # '../outputs/rag_nq_5_cad_Llama-2-13b-chat-hf_short.json',
    # # # '../outputs/rag_nq_5_cad_Llama-2-13b-chat-hf_short_4.json',
    # # # '../outputs/rag_nq_5_cad_Llama-2-13b-chat-hf_short_9.json',
    # # '../outputs/rag_nq_5_cad_Llama-2-7b-chat-hf_short_0_cadall.json',
    # # '../outputs/rag_nq_5_cad_Llama-2-7b-chat-hf_short_4_cadall.json',
    # # '../outputs/rag_nq_5_cad_Llama-2-7b-chat-hf_short_9_cadall.json',
    # '../outputs/rag_nq_1_Llama-2-7b-chat-hf_short_0.json',
    # '../outputs/rag_nq_1_cad_Llama-2-7b-chat-hf_short_0_cadall.json',
    # '../outputs/rag_nq_10_Llama-2-7b-chat-hf_short_0.json',
    # '../outputs/rag_nq_10_Llama-2-7b-chat-hf_short_4.json',
    # '../outputs/rag_nq_10_Llama-2-7b-chat-hf_short_9.json',
    # # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_0.json',
    # # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4.json',
    # # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_9.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall_kl.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall_kl_weight1.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall_kl_weight1_reverse.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall_kl_keep_cad.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall_kl_keep_cad_reverse.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall_kl_keep_cad_fulldiff.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall_kl_keep_cad_fulldiff_sf.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall_ku.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall_ku_reverse.json',
    # '../outputs/rag_nq_10_cad_Llama-2-7b-chat-hf_short_4_cadall_ku_sf.json',
    '../outputs/rag_nq_1_Meta-Llama-3-8B-Instruct_short_0.json',
    '../outputs/rag_nq_10_Meta-Llama-3-8B-Instruct_short_0.json',
    '../outputs/rag_nq_10_Meta-Llama-3-8B-Instruct_short_4.json',
    '../outputs/rag_nq_10_Meta-Llama-3-8B-Instruct_short_9.json',
    '../outputs/rag_nq_10_cad_Meta-Llama-3-8B-Instruct_short_4_cadall_kl_keep_cad_fulldiff_sf.json',
    '../outputs/rag_nq_10_cad_Meta-Llama-3-8B-Instruct_short_4_cadall_kl_keep_cad_fulldiff_sf_reverse.json',
    '../outputs/rag_nq_10_cad_Meta-Llama-3-8B-Instruct_short_4_cadall_ku_sf.json',
    '../outputs/rag_nq_10_cad_Meta-Llama-3-8B-Instruct_short_4_cadall_ku_reverse.json',
    '../outputs/rag_nq_10_Meta-Llama-3-8B-Instruct_short_4_topkportion_sf_0.01_5.json',
    '../outputs/rag_nq_10_Meta-Llama-3-8B-Instruct_short_4_topkportion_sf_0.1_5.json',
    '../outputs/rag_nq_10_Meta-Llama-3-8B-Instruct_short_4_topkportion_sf_0.01_3.json',
]

short_answer_list = []
with gzip.open(data_path, 'rb') as f:
        for line in f:
            data = json.loads(line)
            short_answer_list.append(data['answers'])

for output_path in output_paths:
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
    
    print(round(em_include/len(output) * 100, 2))
