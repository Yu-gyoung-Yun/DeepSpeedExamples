"""
Run OPT with huggingface or deepspeed.

Reference:
https://github.com/FMInference/FlexGen/blob/main/benchmark/hf_ds/hf_opt.py
"""

import argparse
import gc
import multiprocessing as mp
import os

import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from accelerate import init_empty_weights
from timer import timers
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM, 
                          BloomForCausalLM, OPTForCausalLM, LlamaForCausalLM, AutoModelForCausalLM,
                        )
from transformers.deepspeed import HfDeepSpeedConfig
from utils import (GB, add_model_hooks, cache_bytes, disable_torch_init,
                   get_filename, get_quant_config, hidden_bytes, meta_to_cpu,
                   model_bytes, write_benchmark_log)
from packaging import version
from datasets import load_dataset
from tqdm import tqdm
import random
import json
import random
import numpy as np

import sys
sys.path.append("./lm_evaluation_harness")
from lm_evaluation_harness import lm_eval
from lm_evaluation_harness.lm_eval import models as lm_models
from lm_evaluation_harness.lm_eval import tasks, evaluator, utils
from lm_evaluation_harness.lm_eval import evaluator

assert version.parse(deepspeed.__version__) >= version.parse("0.10.3"), "ZeRO-Inference with weight quantization and kv cache offloading is available only in DeepSpeed 0.10.3+, please upgrade DeepSpeed"

def get_model_config(model_name):
    if "175b" in model_name:
        config = AutoConfig.from_pretrained("facebook/opt-66b")
        config.hidden_size = 12288
        config.word_embed_proj_dim = 12288
        config.ffn_dim = 12288 * 4
        config.num_attention_heads = 96
        config.num_hidden_layers = 96
    else:
        config = AutoConfig.from_pretrained(model_name)#, trust_remote_code=True)

    if 'bloom' in model_name:
        config.model_type = 'bloom'

    return config

def get_ds_model(
    model_name,
    dtype,
    cpu_offload,
    disk_offload,
    offload_dir,
    dummy_weights,
    bits,
    group_size,
):

    config = get_model_config(model_name)
    hidden_size = config.hidden_size
    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()
    pin_memory = bool(args.pin_memory)

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size, # 0, 
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
        },
        "overlap_comm": True,
        "steps_per_print": 1,
        "train_batch_size": args.batch_size,
        "train_micro_batch_size_per_gpu": args.batch_size/args.worlds,
        "wall_clock_breakdown": False
    }

    if bits == 4:
        quant_config = get_quant_config(config, bits=bits, group_size=group_size)
        ds_config.update(quant_config)
    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory
        )

    if disk_offload:
        print(f"pin_memory: {pin_memory} w/ {offload_dir}") # True w/ /dev/nvme0
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=True, #pin_memory,
            nvme_path="dev/nvme0", #offload_dir,
            buffer_count=7,
            buffer_size=1e9, #9 * GB if config.model_type == 'bloom' else 2 * GB,
        )
        '''ds_config["aio"] = {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True,
        }'''

    dschf = HfDeepSpeedConfig(
        ds_config
    )  # this tells from_pretrained to instantiate directly on gpus

    # clear cache / free memory
    get_accelerator().empty_cache()
    gc.collect()

    '''if config.model_type in ["bloom", "bloom-7b1"]:
        model = BloomForCausalLM.from_pretrained(
            dummy_weights or model_name, torch_dtype=dtype,
        )
    elif config.model_type == "opt":
        model = OPTForCausalLM.from_pretrained(
            dummy_weights or model_name, torch_dtype=dtype,
        )
    elif config.model_type == "llama":
        print("config.model_type == llama")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
            torch_dtype=torch.float16,
            #model = LlamaForCausalLM.from_pretrained(
            #dummy_weights or model_name, torch_dtype=dtype,
        )
    else:
        raise ValueError(f"Unexpected model type: {config.model_type}")'''

    _model = "hf-causal"
    model_args = f"pretrained={model_name}"
    lm = lm_models.get_model(_model).create_from_arg_string(
        model_args, {"batch_size": args.batch_size, "max_batch_size": None, "device": "cpu"}
    ) # class
    #model = HFLM_class.model
    #model = model.eval()

    #print(f"[BEFORE] model.config = {lm.model.config}") # LlamaForCausalLM
    ds_engine = deepspeed.initialize(model=lm.model, config_params=ds_config)[0]
    ds_engine.module.eval()
    lm.model = ds_engine.module
    print(f"ds_engine: {ds_engine}") # DeepSpeedEngine()
    print(f"model.config = {lm.model.config}")

    return lm, lm.model


def run_generation(
    model_name,
    batch_size,
    prompt_len,
    gen_len,
    cpu_offload,
    disk_offload,
    offload_dir,
    num_nodes,
    num_gpus_per_node,
    dummy,
    output_file,
    verbose,
    kv_offload,
    quant_bits,
    quant_group_size,
    pin_kv_cache,
    async_kv_offload,
):
    # Load tokenizer
    config = get_model_config(model_name)
    return_token_type_ids = False # True 
    padding_side = "left" if config.model_type in ["opt"] else "right"

    if config.model_type == "opt":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name.replace("175b", "66b"), 
            return_token_type_ids=return_token_type_ids,
            padding_side=padding_side,
            use_fast=False, 
        )
    else:  # here
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            return_token_type_ids=return_token_type_ids,
            padding_side=padding_side,
            use_fast=False, 
        )


    tokenizer.pad_token = tokenizer.eos_token

    if hasattr(config, 'torch_dtype'):
        dtype = config.torch_dtype
    else:
        dtype = torch.float

    if dummy:
        filename = os.path.join(
            offload_dir, f"{model_name.replace('/', '-')}-hf-weights/"
        )
        if not os.path.exists(filename):
            print("create dummy weights")
            with init_empty_weights():
                if config.model_type == 'opt':
                    model = OPTForCausalLM(config)
                elif config.model_type in ["bloom", "bloom-7b1"]:
                    model = BloomForCausalLM(config)
                elif config.model_type == "llama":
                    model = AutoModelForCausalLM(config) #LlamaForCausalLM(config)
                else:
                    raise ValueError(f"Unexpected model type: {config.model_type}")                    
            model.save_pretrained(
                filename, state_dict=meta_to_cpu(model.state_dict(), torch.float16)
            )
        dummy_weights = filename
    else:
        dummy_weights = None

    print("load model")
    with torch.no_grad():
        lm, model = get_ds_model(
            model_name,
            dtype,
            cpu_offload,
            disk_offload,
            offload_dir,
            dummy_weights,
            quant_bits,
            quant_group_size,
        )

    # Run generation
    execute_gen_len = gen_len

    # lm-eval
    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    
    random.seed(1234)
    np.random.seed(1234)

    task_dict = lm_eval.tasks.get_task_dict(task_names)
    bootstrap_iters = 100000 # default value in lm_eval
    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    '''#prompts = all_eval_text_to_encode
    prompts = ["Paris is the capital city of"] * (batch_size // dist.get_world_size())

    def _batch_encode(prompts, return_token_type_ids):
        input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding="max_length", max_length=prompt_len, return_token_type_ids=return_token_type_ids)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
        return input_tokens

    input_tokens = _batch_encode(prompts, return_token_type_ids)

    if kv_offload:
        model.set_kv_cache_offload(True, gen_len, pin_kv_cache, async_kv_offload) # AttributeError: 'LlamaForCausalLM' object has no attribute 'set_kv_cache_offload'

    #print(model, model.config)'''


    add_model_hooks(lm.model)
    torch.cuda.set_device(dist.get_rank())
    #timer = timers("generate-forward")
    #timer.start(sync_func=get_accelerator().synchronize)
    results = lm_eval.evaluator.evaluate(
        lm_class=lm,
        lm=lm.model,
        task_dict=task_dict,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        bootstrap_iters=bootstrap_iters,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )
    #timer.stop(sync_func=get_accelerator().synchronize)

    label_map = None
    print("[lm_eval] is over!!")

    def set_model_stage(model, stage):
        model.stage = stage

    # Run
    print(f"benchmark, prompt_len = {prompt_len}, execute_gen_len = {execute_gen_len}, input_ids.shape = {input_tokens.input_ids.shape}")

    generate_kwargs = dict(max_new_tokens=execute_gen_len, do_sample=False)
    prefill_timings = []
    timer = timers("generate-forward")
    for _ in range(2):
        timer.start(sync_func=get_accelerator().synchronize)
        with torch.no_grad():
            set_model_stage(model, "prefill")
            # output_ids = model.generate(input_ids=input_ids, **generate_kwargs)
            output_ids = model.generate(**input_tokens, **generate_kwargs)
            prefill_timings.append(model.__duration__)
        timer.stop(sync_func=get_accelerator().synchronize)
    costs = timers("generate-forward").costs

    if args.local_rank != 0:
        return

    def remove_model_hooks(module):
        if hasattr(module, "__start_time_hook_handle__"):
            module.__start_time_hook_handle__.remove()
            del module.__start_time_hook_handle__
        if hasattr(module, "__end_time_hook_handle__"):
            module.__end_time_hook_handle__.remove()
            del module.__end_time_hook_handle__
        if hasattr(module, "stage"):
            del module.stage
        if hasattr(module, "__duration__"):
            del module.__duration__

    # Log output
    print(f"Summary:")
    print(f"costs = {costs}, prefill_timings = {prefill_timings}")
    total_latency = costs[-1]
    prefill_latency = prefill_timings[-1]
    remove_model_hooks(model)

    prefill_throughput = batch_size * prompt_len / prefill_latency
    decode_latency = total_latency - prefill_latency
    decode_throughput = batch_size * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = batch_size * gen_len
    total_throughput = num_generated_tokens / total_latency
    gpu_peak_mem = get_accelerator().max_memory_allocated(torch.device("cuda"))
    out_str = ""

    if verbose >= 2:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(f"outputs: {len(outputs)}")
        show_str = "Outputs:\n" + 70 * "-" + "\n"
        for i in [0, (len(outputs) - 1) // 2, len(outputs) - 1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += 70 * "-" + "\n"
        print(show_str)

        # Check lengths
        input_lens = [len(x) for x in input_tokens.input_ids]
        output_lens = [len(x) for x in output_ids]
        assert all(x == prompt_len for x in input_lens)
        assert all(x == prompt_len + execute_gen_len for x in output_lens)

    if output_file == "auto":
        filename = (
            get_filename(
                model_name,
                batch_size,
                prompt_len,
                gen_len,
                cpu_offload,
                disk_offload,
                num_nodes,
                num_gpus_per_node,
                kv_offload,
                quant_bits != 16,
            )
            + ".log"
        )
    else:
        filename = output_file

    cache_size = cache_bytes(config, batch_size, prompt_len + gen_len)
    hidden_size = hidden_bytes(config, batch_size, prompt_len + gen_len)
    log_str = write_benchmark_log(
        filename,
        model_bytes(config),
        cache_size,
        hidden_size,
        gpu_peak_mem,
        prefill_latency,
        prefill_throughput,
        decode_latency,
        decode_throughput,
        total_latency,
        total_throughput,
    )
    if verbose >= 1:
        print(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="model name or path; currently only supports OPT and BLOOM models")
    parser.add_argument("--dummy", action="store_true", help="Use dummy weights for benchmark purposes.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=512,  help="prompt length")
    parser.add_argument("--gen-len", type=int, default=32,  help="number of tokens to generate")
    parser.add_argument("--local_rank", type=int, help="local rank for distributed inference")
    parser.add_argument("--pin-memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--cpu-offload", action="store_true", help="Use cpu offload.")
    parser.add_argument("--disk-offload", action="store_true", help="Use disk offload.")
    parser.add_argument("--offload-dir", type=str, default="~/offload_dir", help="Directory to store offloaded cache.")
    parser.add_argument("--kv-offload", action="store_true", help="Use kv cache cpu offloading.")
    parser.add_argument("--log-file", type=str, default="auto", help="log file name")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    parser.add_argument("--pin_kv_cache", action="store_true", help="Allocate kv cache in pinned memory for offloading.")
    parser.add_argument("--async_kv_offload", action="store_true", help="Using non_blocking copy for kv cache offloading.")
    parser.add_argument("--output_dir", type=str, default="./data", help="enable hybrid engine testing")
    parser.add_argument("--seed", type=int, default=42, help="enable hybrid engine testing")
    parser.add_argument("--worlds", type=int, default=2, help="number of total gpus in program")
    parser.add_argument("--tasks", default="hellaswag", choices=lm_eval.utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=1000,
                    help="Limit the number of examples per task. "
                            "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--pin_memory", action="store_true", help="Using non_blocking copy for kv cache offloading.")
    #parser.add_argument("--model_args", default="")

    args = parser.parse_args()

    deepspeed.init_distributed()    
    num_gpus_per_node = get_accelerator().device_count()
    num_nodes = dist.get_world_size() // num_gpus_per_node


    run_generation(
        args.model,
        args.batch_size,
        args.prompt_len,
        args.gen_len,
        args.cpu_offload,
        args.disk_offload,
        os.path.abspath(os.path.expanduser(args.offload_dir)),
        num_nodes,
        num_gpus_per_node,
        args.dummy,
        args.log_file,
        args.verbose,
        args.kv_offload,
        args.quant_bits,
        args.quant_group_size,
        args.pin_kv_cache,
        args.async_kv_offload,
    )
