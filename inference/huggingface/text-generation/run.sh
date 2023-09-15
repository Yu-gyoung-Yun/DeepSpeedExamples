deepspeed --num_gpus 4 inference-test.py \
    --checkpoint_path ./LLaMA-7B-Official --batch_size 2 --model /NAS/JG/llama_hf/7B  \
    --use_meta_tensor --dtype float16
