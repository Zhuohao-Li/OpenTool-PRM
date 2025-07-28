NUM_GPUS=8
for (( i=0; i<NUM_GPUS; i++ )); do
    echo "Starting server on port $((7000+i)) with GPU: $i"
    CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
        --model-path Skywork/Skywork-Reward-V2-Llama-3.1-8B \
        --mem-fraction-static 0.9 \
        --tp 1 \
        --host 127.0.0.1 \
        --port $((8000+i)) \
        --context-length 16384 \
        --is-embedding \
        &
done