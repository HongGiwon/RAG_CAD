import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Any, List
from datasets import load_dataset
from tqdm import tqdm
import pickle
import json

def get_ll2_model(model_name: str, dtype: torch.dtype = torch.bfloat16) -> Tuple[Any, Any]:
    model_kwargs = {}
    model_kwargs['torch_dtype'] = dtype
    model_kwargs['device_map'] = 'auto'
    access_token = "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"

    tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True, padding_side="left",cache_dir="/mnt/ceph_rbd/mistral7b/pretrained/", use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, resume_download=True, trust_remote_code=True,cache_dir="/mnt/ceph_rbd/mistral7b/pretrained/", use_auth_token=access_token, **model_kwargs)

    model.eval()
    return tokenizer, model

def next_single_tok_gen(prompt, generated_token_accum, max_ans_len):
    encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt", max_length=max_seq_len)
    encodeds = torch.cat([encodeds, generated_token_accum], dim=1)
    model_inputs = encodeds.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(model_inputs, max_new_tokens=max_ans_len, do_sample=False, return_dict_in_generate=True, output_scores = True, temperature=None, top_p=None)
    return generated_ids, len(model_inputs[0]) - generated_token_accum[0].shape[0]

def next_full_tok_gen(prompt, generated_token_accum, max_ans_len):
    encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt", max_length=max_seq_len)
    encodeds = torch.cat([encodeds, generated_token_accum], dim=1)
    model_inputs = encodeds.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(model_inputs, max_new_tokens=max_ans_len, do_sample=True)
    return generated_ids, len(model_inputs[0]) - generated_token_accum[0].shape[0]

data_path = "prompts/rag_nq_5_cad_chat.json"
full_data_path = "prompts/rag_nq_5_chat.json"
output_path = "outputs/rag_nq_5_cad_chat_ll2_7b.json"
model_name = "meta-llama/Llama-2-7b-chat-hf"

num_retrieved_docs = 5
max_seq_len = 4096
max_ans_len_total = 64
max_ans_len = 1
alpha = 0.5

tokenizer, model = get_ll2_model(model_name)
device = "cuda"

with open(data_path, 'r') as f:
    input_prompts = json.load(f)
with open(full_data_path, 'r') as f:
    full_input_prompts = json.load(f)

output_prompts = []

for cad_prompts, full_prompt in tqdm(zip(input_prompts, full_input_prompts)):
    generated_token_accum = torch.tensor([[29871]]) # space for ll2
    score_list = []

    prompt, score = cad_prompts[-1] # empty
    empty_generated_ids, _ = next_single_tok_gen(prompt, generated_token_accum, max_ans_len=max_ans_len)
    empty_score = empty_generated_ids['scores'][0][0].cpu()

    for prompt, score in cad_prompts[:num_retrieved_docs]:
        generated_ids, _ = next_single_tok_gen(prompt, generated_token_accum, max_ans_len=max_ans_len)
        cad_score = generated_ids['scores'][0][0].cpu()
        score_list.append((1+alpha) * cad_score - alpha * empty_score)

    stacked_tensors = torch.stack(score_list)
    score_mean = torch.mean(stacked_tensors, dim=0)

    values, indices = torch.topk(score_mean, 1)
    generated_token_accum = torch.cat([generated_token_accum,indices.unsqueeze(dim=0)], dim=1)

    ### full prompt + generated token
    generated_ids, new_tok_idx = next_full_tok_gen(full_prompt, generated_token_accum, max_ans_len=max_ans_len_total-max_ans_len)
    generated_ans = tokenizer.decode(generated_ids[0][new_tok_idx:], skip_special_tokens=False)
    output_prompts.append(generated_ans)

with open(output_path, 'w') as f:
    json.dump(output_prompts, f)