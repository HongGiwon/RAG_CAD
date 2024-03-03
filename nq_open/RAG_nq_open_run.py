import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Any, List
from datasets import load_dataset
from tqdm import tqdm
import pickle
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-70b-chat-hf",
                        help="Name of pretrained model")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="max seq len")
    parser.add_argument("--max_gen_len", type=int, default=64, help="max gen len")
    parser.add_argument("--num_retrieved_docs", type=int, default=5, help="num_of_retrieved_docs")

    args = parser.parse_args()
    return args

def get_ll2_model(model_name: str, dtype: torch.dtype = torch.bfloat16) -> Tuple[Any, Any]:
    model_kwargs = {}
    model_kwargs['torch_dtype'] = dtype
    model_kwargs['device_map'] = 'auto'
    access_token = "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"

    tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True, padding_side="left",cache_dir="/mnt/ceph_rbd/mistral7b/pretrained/", use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, resume_download=True, trust_remote_code=True,cache_dir="/mnt/ceph_rbd/mistral7b/pretrained/", use_auth_token=access_token, **model_kwargs)

    model.eval()
    return tokenizer, model

if __name__ == "__main__":
    args = parse_args()

    num_retrieved_docs = 10
    data_path = "prompts/rag_nq_" +str(num_retrieved_docs)+ "_chat.json"
    output_path = "outputs/rag_nq_" +str(num_retrieved_docs)+ "_chat_ll2_7b.json"
    model_name = args.model_name
    
    max_seq_len = args.max_seq_len
    max_ans_len = args.max_gen_len
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