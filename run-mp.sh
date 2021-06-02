# Change for multinode config
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    main.py \
    --use_gpu \
    --model "baseline" \
    --model_path "squad_model.pt" \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path   "datasets/squad_dev.jsonl.gz" \
    --output_path "squad_predictions.txt" \
    --rnn_cell_type "lstm" \
    --hidden_dim 256 \
    --bidirectional \
    --dropout 0.3 \
    --learning_rate 5e-4 \
    --early_stop 3 \
    --shuffle_examples \
    --do_train \
    --do_test
