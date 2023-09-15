 deepspeed --num_gpus 4 run_model.py --model /NAS/JG/llama_hf/7B --batch-size 1 --prompt-len 256 --gen-len 32 --cpu-offload --worlds 4
