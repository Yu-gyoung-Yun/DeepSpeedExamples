from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed
import math
import os
import torch
import time
from utils import DSPipeline, Performance
from deepspeed.runtime.utils import see_memory_usage
from arguments import parser
from datasets import load_dataset
from tqdm import tqdm
import random
import json
args = parser.parse_args()
'''deepspeed --num_gpus 4 inference-test.py \
    --checkpoint_path ./LLaMA-7B-Official --batch_size 2 --model /NAS/JG/llama_hf/7B  \
    --use_meta_tensor --dtype float16'''
if args.hf_baseline and args.world_size > 1:
    raise RuntimeError("Only `--num_gpus 1` supported for non-DeepSpeed uses")

data_type = getattr(torch, args.dtype)
print(f"data_type: {data_type}") # torch.float16
if args.local_rank == 0:
    see_memory_usage("before init", True)

t0 = time.time()

pipe = DSPipeline(model_name=args.model,
                  dtype=data_type,
                  is_meta=args.use_meta_tensor,
                  device=args.local_rank,
                  checkpoint_path=args.checkpoint_path)

if args.local_rank == 0:
    print(f"initialization time: {(time.time()-t0) * 1000}ms")
    see_memory_usage("after init", True)

if args.use_meta_tensor:
    ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)
else:
    ds_kwargs = dict()

# Use DeepSpeed Hybrid Engine for inference
if args.test_hybrid_engine:
    ds_config = {"train_batch_size": args.batch_size, "fp16": {"enabled": True if data_type==torch.half else False}, "hybrid_engine": {"enabled": True}}
    pipe.model, *_ = deepspeed.initialize(model=pipe.model, config=ds_config)
    pipe.model.eval()
# If not trying with the HuggingFace baseline, use DeepSpeed Inference Engine
else:
    if not args.hf_baseline:
        pipe.model = deepspeed.init_inference(pipe.model,
                                    dtype=data_type,
                                    mp_size=args.world_size,
                                    replace_with_kernel_inject=args.use_kernel,
                                    max_tokens=args.max_tokens,
                                    save_mp_checkpoint_path=args.save_mp_checkpoint_path,
                                    **ds_kwargs
                                    )
  
if args.local_rank == 0:
    see_memory_usage("after init_inference", True)

def process_hellaswag_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process hellaswag examples'):
        processed_examples.append({
            'id': idx,
            'ctx_a': raw_data['ctx_a'],
            'ctx_b': raw_data['ctx_b'],
            'ctx':raw_data['ctx'],
            'endings':raw_data['endings'],
            'label':int(raw_data['label']),
            'activity_label':raw_data['activity_label']
        })
        idx += 1
    return processed_examples

task_name='hellaswag'
'''if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
        os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
    print('use cached examples')
    with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
        total_train_examples = json.load(f)
    with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
        total_eval_examples = json.load(f)
else:'''
print("[LOAD] dataset")
hellaswag_datasets = load_dataset('hellaswag',cache_dir="./cache_dir")
total_eval_examples = [e for e in hellaswag_datasets['validation']]
total_eval_examples = random.sample(total_eval_examples, 256)
total_eval_examples = process_hellaswag_examples(total_eval_examples)
with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
    json.dump(total_eval_examples, f, indent=4)
'''if args.debug:
    args.annotation_size = 10
    args.batch_size = 1
    total_train_examples = total_train_examples[:50]
    total_eval_examples = total_eval_examples[:5]
def format_example(example,label_map,**kwargs):
    return f"The topic is {example['activity_label']}. {example['ctx_a']} " \
            f"{example['ctx_b']} ",f"{example['endings'][example['label']]}"'''

all_eval_text_to_encode = [f"The topic is {raw_item['activity_label']}. {raw_item['ctx_a']} {raw_item['ctx_b']} | " \
                            f"{raw_item['endings'][0]} | " \
                            f"{raw_item['endings'][1]} | " \
                            f"{raw_item['endings'][2]} | " \
                            f"{raw_item['endings'][3]}" for raw_item in total_eval_examples]
label_map = None

input_sentences = [
         "DeepSpeed is a machine learning framework",
         "He is working on",
         "He has a",
         "He got all",
         "Everyone is happy and I can",
         "The new movie that got Oscar this year",
         "In the far far distance from our galaxy,",
         "Peace is the only way"
]

input_sentences = all_eval_text_to_encode

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]

iters = 30 if args.test_performance else 2 #warmup
times = []
for i in range(iters):
    torch.cuda.synchronize()
    start = time.time()
    outputs = pipe(inputs,
            num_tokens=args.max_new_tokens,
            do_sample=(not args.greedy))
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)
print(f"generation time is {times[1]} sec")

if args.local_rank == 0:
    for i, o in zip(inputs, outputs):
        print(f"\nin={i}\nout={o}\n{'-'*60}")
    if args.test_performance:
        Performance.print_perf_stats(map(lambda t: t / args.max_new_tokens, times), pipe.model.config, args.dtype, args.batch_size)

