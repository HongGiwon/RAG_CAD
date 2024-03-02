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

data_path = "prompts/rag_nq_5_chat.json"
output_path = "outputs/rag_nq_5_chat_ll2_7b.json"
model_name = "meta-llama/Llama-2-7b-chat-hf"
max_seq_len = 4096
max_ans_len = 64

tokenizer, model = get_ll2_model(model_name)
device = "cuda"

with open(data_path, 'r') as f:
    input_prompts = json.load(f)

output_prompts = []
for prompt in tqdm(input_prompts):
    encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt", max_length=max_seq_len)
    model_inputs = encodeds.to(device)
    
    with torch.inference_mode():
        generated_ids = model.generate(model_inputs, max_new_tokens=max_ans_len, do_sample=True)
    
    generated_ans = tokenizer.decode(generated_ids[0][len(model_inputs[0]):], skip_special_tokens=True)
    output_prompts.append(generated_ans)
    
with open(output_path, 'w') as f:
    json.dump(output_prompts, f)