import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Any, List
from datasets import load_dataset
from tqdm import tqdm
import pickle
import json
import argparse

import torch

def kurtosis(x):
    mean = x.mean()
    deviations = x - mean
    var = torch.mean(deviations ** 2)
    std = torch.sqrt(var)
    
    kurt = torch.mean((deviations / std) ** 4) - 3
    return kurt + 3

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-70b-chat-hf",
                        help="Name of pretrained model")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="max seq len")
    parser.add_argument("--max_gen_len", type=int, default=32, help="max gen len")
    parser.add_argument("--num_retrieved_docs", type=int, default=5, help="num_of_retrieved_docs")
    parser.add_argument("--ans_pos", type=int, default=0, help="ans_pos")
    parser.add_argument("--alpha", type=float, default=0.5, help="cad alpha")

    args = parser.parse_args()
    return args

def get_ll2_model(model_name: str, dtype: torch.dtype = torch.bfloat16) -> Tuple[Any, Any]:
    model_kwargs = {}
    model_kwargs['torch_dtype'] = dtype
    model_kwargs['device_map'] = 'auto'
    access_token = "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"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/mnt/ceph_rbd/mistral7b/pretrained/", use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/mnt/ceph_rbd/mistral7b/pretrained/", use_auth_token=access_token, **model_kwargs)

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

if __name__ == "__main__":
    args = parse_args()

    num_retrieved_docs = args.num_retrieved_docs
    ans_pos = args.ans_pos

    data_path = "prompts/rag_nq_" +str(num_retrieved_docs)+ "_cad_chat_short_" + str(ans_pos) + ".json"
    full_data_path = "prompts/rag_nq_" +str(num_retrieved_docs)+ "_chat_short_" + str(ans_pos) + ".json"
    output_path = "outputs/rag_nq_" +str(num_retrieved_docs)+ "_cad_" + args.model_name.split("/")[-1] + "_short_" + str(ans_pos) + "_cadall_ku_reverse.json"
    model_name = args.model_name

    max_seq_len = args.max_seq_len
    max_ans_len_total = args.max_gen_len
    max_ans_len = 1
    alpha = args.alpha

    tokenizer, model = get_ll2_model(model_name)
    #device = "cuda"
    device = model.device
    space_tok_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(": "))[-1]
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    with open(data_path, 'r') as f:
        input_prompts = json.load(f)
    with open(full_data_path, 'r') as f:
        full_input_prompts = json.load(f)

    output_prompts = []

    softmax_weight = torch.nn.Softmax(dim=0)
    for cad_prompts, full_prompt in tqdm(zip(input_prompts, full_input_prompts)):
        generated_token_accum = torch.tensor([[space_tok_id]]) # space for ll2

        empty_prompt, score = cad_prompts[-1] # empty

        for iter_gen in range(max_ans_len_total):
            empty_generated_ids, _ = next_single_tok_gen(empty_prompt, generated_token_accum, max_ans_len=max_ans_len)
            empty_score = softmax_weight(empty_generated_ids['scores'][0][0].cpu())
            score_list = []
        
            for prompt, score in cad_prompts[:num_retrieved_docs]:
                generated_ids, new_tok_idx = next_single_tok_gen(prompt, generated_token_accum, max_ans_len=max_ans_len)
                cad_score = generated_ids['scores'][0][0].cpu()
                score_list.append(softmax_weight(cad_score))

            #stacked_tensors = torch.stack(score_list)
            
            kurtosis_scores = torch.tensor([kurtosis(tensor) for tensor in score_list])
            weights = kurtosis_scores / kurtosis_scores.sum()

            weighted_sum = torch.zeros_like(empty_score)
            for tensor, weight in zip(score_list, weights):
                weighted_sum += tensor * (1/weight)
            
            values, indices = torch.topk(weighted_sum, 1)
            generated_token_accum = torch.cat([generated_token_accum,indices.unsqueeze(dim=0)], dim=1)

        generated_ans = tokenizer.decode(generated_token_accum[0], skip_special_tokens=False)
        output_prompts.append(generated_ans)

    with open(output_path, 'w') as f:
        json.dump(output_prompts, f)
